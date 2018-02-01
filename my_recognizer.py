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
            
