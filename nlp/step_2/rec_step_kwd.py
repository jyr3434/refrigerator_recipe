from konlpy.tag import Okt,Mecab
from konlpy.corpus import kobill
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re,time
from multiprocessing import Pool




class konlpy:

    def __init__(self):
        self.okt = None

    def init_okt(self):
        self.okt = Okt()

    def rec_step_tokenize(self, iterrow):

        token_set = set()
        sentence_list = iterrow[1].rec_step.split('|')
        for sentence in sentence_list:
            token_list = self.okt.nouns(sentence)
            # print(token_list,type(token_list))
            token_set.update(token_list)
        return token_set

    def Check_token(self,list1, list2):
        ########################################
        # 재료 토큰, 순서 토큰 내부 비교
        n = 0
        tot = []
        for i in range(0, len(list1)):
            not_subset_token = []
            for x in list1[i]:
                if x not in list2[i]:
                    # print(x)
                    not_subset_token.append(x)
            if not_subset_token: n+=1
            tot.append(not_subset_token)
        # for l in tot:
        #     print(l)
        # print(n)
        return tot, n
        #########################################
okt = None

def multiprocessing_initializer():
    global okt
    okt = Okt()

def rec_step_tokenize(iterrow):
    token_row = ''
    id = iterrow[1].id
    sentence_list = iterrow[1].rec_step.split('|')
    for sentence in sentence_list:
        token_list = okt.nouns(sentence)
        # print(token_list,type(token_list))
        # token_list = [i[0] for i in token_list if i[1] in ('Noun','Verb')]
        token_row += '|'.join(token_list)
    return id,token_row


if __name__ == '__main__':
    start_time = time.time()

    filename1 = '../../../data/crawl_data/recipe_data_dropna.csv'
    data1 = pd.read_csv(filename1, encoding='UTF-8')
    data_source = data1.loc[:, ('id','rec_step')]
    # print(data_source)



    '''
    kon = Konlpy()
    all_token_set = set()
    if isinstance(data_source,pd.DataFrame):
        for row in data_source.iterrows():
            print(row)
            all_token_set.update(kon.rec_step_tokenize(row))
        df_data = sorted(list(all_token_set))
        print(df_data)
        pd.DataFrame(df_data).to_csv('../../../data/nlp_data/kwd_step.csv')
    print("--- %.4f seconds ---" % (time.time() - start_time))
    '''
    all_token_set = set()
    pool = Pool(processes=8,initializer=multiprocessing_initializer)
    tokens_l = pool.map(rec_step_tokenize,data_source.iterrows())
    id = [i[0] for i in tokens_l]
    step = [i[1] for i in tokens_l]
    pd.DataFrame({'id':id,'rec_step':step}).to_csv('../../../data/nlp_data/kwd_step_noun_okt.csv')
    # _pickle.PicklingError: Can
    # 't pickle <java class '
    # kr.lucypark.okt.OktInterface
    #'>: attribute lookup kr.lucypark.okt.OktInterface on jpype._jclass failed
    print("--- %.4f seconds ---" % (time.time() - start_time))
