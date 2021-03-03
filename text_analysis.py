import commons
import ml_commons
import logging
import sys
from semantic_bert.models import WebBertSimilarity
from text_sim_predict import TextSim
from seq_matcher.sequence import SimilarityRatio
from topic_modelling.topic_analysis import TopicModel
from syntax_analysis.syntactic import NLPStanzaAnalysis
from services.text_sim_service import TextSimService
from entity.spell_model.spell_checker import SpellCheckModel
from preprocessor.clean import Clean


logger = logging.getLogger(__name__)
nlp_s = NLPStanzaAnalysis()
similiarity = SimilarityRatio()
spell_checker = SpellCheckModel()
web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction

class TextAnalysis():

    def __init__(self):
        self.text_sim = TextSimService()

    def preprocessing(sentence1,sentence2):
        token1, token2 = Clean.preprocessing(sentence1),Clean.preprocessing(sentence2)
        logger.info(Clean.preprocessing(sentence1),Clean.preprocessing(sentence2))
        logger.info(Clean.clean_text(Clean.normalize(token1)))
        sentence1,sentence2 = Clean.clean_text(Clean.normalize(token1)),Clean.clean_text(Clean.normalize(token1))
        logger.info("spell model output: {}".format(spell_checker.spell_corrector(sentence1)))
        logger.info("spell model output: {}".format(spell_checker.spell_corrector(sentence2)))
        sentence1,sentence2 = spell_checker.spell_corrector(sentence1),spell_checker.spell_corrector(sentence2)
        return sentence1,sentence2 

    def text_analysis(self,sentence1,sentence2):
        #hardcoded value for table------------
        if sentence1.lower() != sentence2.lower():
            preprocessing = True
            org_id = "scanta"
            customer_id = "hello@scanta.io"
            conversation_id = {'246gvfy-ur4900b4-hui2r2w','af46gvfy-r4327vgb4-hvfh57'}
            service_id = {'1','2'}
            compared_conv = {sentence1,sentence2}
            #--------------------------------------
            label_flag = ''
            sentence1,sentence2 = TextAnalysis.preprocessing(sentence1,sentence2)
            textsim = TextSim(sentence1,sentence2)
            is_similar = textsim.predict_score()
            bert_sim = web_model.predict([(sentence1,sentence2)])
            score_nn = is_similar[0][2]*100
            score_bert = (bert_sim*100)/5
            if score_nn > 50.0 and score_bert[0] > 50.00:
                label_flag = 'is_similar'
            else:
                label_flag = 'not_similar' 
            logger.info("Semantic Analysis---Score: nn: {},bert: {} and flag assigned {}".format(score_nn,score_bert[0],label_flag))
            seq_ratio = similiarity.similarity_check(sentence1,sentence2)*100
            logger.info("Sequence Match Analysis---Sequence Match Ratio : ",seq_ratio)
            syntax1 = nlp_s.percentage_analysis(sentence1)
            syntax2 = nlp_s.percentage_analysis(sentence2)
            syntax_res = {sentence1 : syntax1, sentence2 : syntax2}
            logger.info("Syntax Analysis: ",syntax_res)
            topic1 = TopicModel.topic_modelling(sentence1)
            topic2 = TopicModel.topic_modelling(sentence2)
            topics_res = {'topics':list(set(topic1[0]+topic2[0])),'topic_distance':{**topic1[1], **topic2[1]}}
            logger.info("Topic Modelling: ",topics_res)
            score = {score_nn,score_bert[0]}
            self.text_sim.add_text_sim(org_id,customer_id,service_id,conversation_id,compared_conv,preprocessing,seq_ratio,score,syntax_res,label_flag)
            return sentence1,sentence2,score_nn,score_bert[0],label_flag,seq_ratio,syntax_res,topics_res
        else:
            return "Similar Sentences compared"

if __name__ == "__main__":
    sentence1,sentence2 = 'What can make python easy to learn?','How can you make python easy to learn?'
    print(TextAnalysis.text_analysis(sentence1,sentence2))



