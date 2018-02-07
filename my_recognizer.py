import warnings
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
    for test_word_id, test_word_XLengths in test_all_XLengths.items():
        prob_dict = {}
        test_word_X, test_word_lengths = test_word_XLengths
        for model_word, model in models.items():
            try:
                score = model.score(test_word_X, test_word_lengths)
            except:
                pass
            prob_dict[model_word] = score
        probabilities.append(prob_dict)
        best_guess_word = max(prob_dict, key=lambda k: prob_dict[k])
        guesses.append(best_guess_word)
        
    return probabilities, guesses

def log_lm(lm, previous_word, current_word, lm_ration):
    new_previous_word = previous_word
    new_current_word = current_word
    if new_previous_word[-1].isdigit:
        new_previous_word = new_previous_word[:-1]
    if new_current_word[-1].isdigit:
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
    return lm_ration * lm.log_p(two_gram)

def recognize_two_gram(lm, probabilities: list, test_set: SinglesData, lm_ratio: float):
    S = 0
    N = len(test_set.wordlist)
    for video_num in test_set.sentences_index:
        possible_sentence_words = [[('<s>', '', 0.0)]]
        for word_index in test_set.sentences_index[video_num]:
            top_word_probabilities = []
            for possible_word, logValue in probabilities[word_index].items():
                for previous_word,_,_ in possible_sentence_words[-1]:
                    lm_logValue = log_lm(lm, previous_word, possible_word, lm_ratio)
                    total_logValue = lm_logValue + logValue
                    if len(top_word_probabilities) < 3:
                        top_word_probabilities.append((possible_word, previous_word, total_logValue))
                        top_word_probabilities.sort(key=lambda tup: tup[2])
                    else:
                        for i in range(len(top_word_probabilities)):
                            if top_word_probabilities[i][2] < total_logValue:
                                top_word_probabilities[i] = (possible_word, previous_word, total_logValue)
                                top_word_probabilities.sort(key=lambda tup: tup[2])
                                break
            possible_sentence_words.append(top_word_probabilities)
            
        reversed_prob_sentences = []
        for last in possible_sentence_words[-1]:
            reversed_prob_sentence = []
            sentence_prob = log_lm(lm, last[0], '</s>', lm_ratio)
            for i in range(len(possible_sentence_words) - 1):
                index = len(possible_sentence_words) - 1 - i
                if i == 0:
                    reversed_prob_sentence.append(last[0])
                    previous_word = last[1]
                    sentence_prob += last[2]
                else:
                    current_possibilities = possible_sentence_words[index]
                    for possible in current_possibilities:
                        if possible[0] == previous_word:
                            reversed_prob_sentence.append(possible[0])
                            previous_word = possible[1]
                            sentence_prob += possible[2]
                            break
            reversed_prob_sentences.append((reversed_prob_sentence, sentence_prob))
        reversed_recognized_sentence, _ = max(reversed_prob_sentences, key = lambda tup: tup[1])
            
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
            
