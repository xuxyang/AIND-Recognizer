import warnings
import math
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    test_all_XLengths = test_set.get_all_Xlengths()
    for test_word_X, test_word_lengths in test_all_XLengths.values():
        prob_dict = {}
        best_guess_word, highest_score = None, -math.inf
        for model_word, model in models.items():
            try:
                score = model.score(test_word_X, test_word_lengths)
                if score > highest_score:
                    best_guess_word, highest_score = model_word, score
            except:
                score = -math.inf
            prob_dict[model_word] = score
        probabilities.append(prob_dict)
        guesses.append(best_guess_word)
        
    return probabilities, guesses

def log_lm(lm, previous_word, current_word, lm_ratio):
    new_previous_word = previous_word
    new_current_word = current_word
    if new_previous_word[-1].isdigit():
        new_previous_word = new_previous_word[:-1]
    if new_current_word[-1].isdigit():
        new_current_word = new_current_word[:-1]
    try:
        lm.p(new_current_word)
    except KeyError:
        new_current_word = '[UNKNOWN]'
    try:
        lm.p(new_previous_word)
    except KeyError:
        new_previous_word = '[UNKNOWN]'
    two_gram = '{} {}'.format(new_previous_word, new_current_word)
    return lm_ratio * lm.log_p(two_gram)

def find_top_word_probabilities(probabilities, word_index, possible_sentence_words, lm, lm_ratio):
    top_word_probabilities = []
    for possible_word, hmm_logValue in probabilities[word_index].items():
        best_previous_word, best_total_two_gram_logValue = '', -math.inf
        for previous_word,_,_ in possible_sentence_words[-1]:
            lm_logValue = log_lm(lm, previous_word, possible_word, lm_ratio)
            total_two_gram_logValue = lm_logValue + hmm_logValue
            if best_total_two_gram_logValue < total_two_gram_logValue:
                best_previous_word, best_total_two_gram_logValue = previous_word, total_two_gram_logValue
        if len(top_word_probabilities) < 5:
            top_word_probabilities.append((possible_word, best_total_two_gram_logValue, hmm_logValue))
            top_word_probabilities.sort(key=lambda tup: tup[1])
        else:
            for i in range(len(top_word_probabilities)):
                if top_word_probabilities[i][1] < best_total_two_gram_logValue:
                    top_word_probabilities[i] = (possible_word, best_total_two_gram_logValue, hmm_logValue)
                    top_word_probabilities.sort(key=lambda tup: tup[1])
                    break
    return top_word_probabilities

def best_previousWord_sentenceLog(lm, possible_sentence_words, hmm_logValue, current_possible_word, lm_ratio):
    max_total_sentence_logValue, best_previous_word = -math.inf, ''
    for previous_possible_word, previous_previous_word, previous_total_logValue in possible_sentence_words[-1]:
        total_sentence_logValue = previous_total_logValue + hmm_logValue + log_lm(lm, previous_possible_word, current_possible_word, lm_ratio)
        if max_total_sentence_logValue < total_sentence_logValue:
            max_total_sentence_logValue, best_previous_word = total_sentence_logValue, previous_possible_word
    return max_total_sentence_logValue, best_previous_word

def search_highlikely_sentence_words(lm, probabilities, test_set, lm_ratio, video_num):
    possible_sentence_words = [[('<s>', '', 0.0)]]
    for word_index in test_set.sentences_index[video_num]:
        top_word_probabilities = find_top_word_probabilities(probabilities, word_index, possible_sentence_words, lm, lm_ratio)
        current_possibles = []
        for current_possible_word,_,hmm_logValue in top_word_probabilities:
            max_total_sentence_logValue, best_previous_word = best_previousWord_sentenceLog(lm, possible_sentence_words, hmm_logValue, current_possible_word, lm_ratio)
            current_possibles.append((current_possible_word, best_previous_word, max_total_sentence_logValue))
        possible_sentence_words.append(current_possibles)

    max_total_sentence_logValue, best_last_word = best_previousWord_sentenceLog(lm, possible_sentence_words, 0.0, '</s>', lm_ratio)    
    possible_sentence_words.append([('</s>', best_last_word, max_total_sentence_logValue)])
    return possible_sentence_words

def find_mostlikely_reversed_sentence(possible_sentence_words):
    reversed_recognized_sentence = []
    for i in range(len(possible_sentence_words) - 2):
        index = len(possible_sentence_words) - 1 - i
        if i == 0:
            reversed_recognized_sentence.append(possible_sentence_words[index][0][1])
        else:
            last_recognized_word = reversed_recognized_sentence[-1]
            for word,previous_word,_ in possible_sentence_words[index]:
                if word == last_recognized_word:
                    reversed_recognized_sentence.append(previous_word)
    return reversed_recognized_sentence

def recognize_two_gram(lm, probabilities: list, test_set: SinglesData, lm_ratio: float):
    S = 0
    N = len(test_set.wordlist)
    for video_num in test_set.sentences_index:
        possible_sentence_words = search_highlikely_sentence_words(lm, probabilities, test_set, lm_ratio, video_num)
        
        reversed_recognized_sentence = find_mostlikely_reversed_sentence(possible_sentence_words)
        
        correct_sentence = [test_set.wordlist[i] for i in test_set.sentences_index[video_num]]
        recognized_sentence = []
        for i in range(len(reversed_recognized_sentence)):
            index = len(reversed_recognized_sentence) - 1 - i
            if reversed_recognized_sentence[index] != correct_sentence[i]:
                recognized_sentence.append('*' + reversed_recognized_sentence[index])
                S += 1
            else:
                recognized_sentence.append(reversed_recognized_sentence[index])
        print('{:5}: {:60}  {}'.format(video_num, ' '.join(recognized_sentence), ' '.join(correct_sentence)))
        
    print("\n**** WER = {}".format(float(S) / float(N)))
    print("Total correct: {} out of {}".format(N - S, N))
            
