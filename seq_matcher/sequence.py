from difflib import SequenceMatcher
import logging
logger = logging.getLogger(__name__)


# check all the sentences for similarilty index wrt standard compairision

class SimilarityRatio():
    
    def get_similarity(self,input_str, standard):
        '''Returns a similarity index using difflib Sequence Matcher function.
        Input: sentence to be compared & standard sentence
        Return : similarity ration in range 0-1. '''
        return SequenceMatcher(None, input_str, standard).ratio()

    def similarity_check(self,input_str, standard):
        dict_ratio = self.get_similarity(input_str, standard)
        return dict_ratio

if __name__=='__main__':
    all_sentences = 'show my credit card balance'
    comparison_standard = 'my balance for credit card.'
    similiarity = SimilarityRatio()
    print(similiarity.similarity_check(all_sentences,comparison_standard))