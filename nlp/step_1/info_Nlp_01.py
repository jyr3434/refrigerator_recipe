from konlpy.tag import Okt,Mecab
from konlpy.corpus import kobill
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re,time

okt = Okt()
#mec = Mecab()

# stopfile = ''
# stopword = open(stopfile).read()

start_time = time.time()

filename1 = './crawl_data/recipe_data_dropna.csv'
#filename2 = './crawl_data/id_4category.csv'


data1 = pd.read_csv(filename1, encoding='UTF-8', index_col=0).head(10000).replace(',',' ')
#data2 = pd.read_csv(filename2, encoding='UTF-8', index_col=0).head(10)

mycol = ['recipe_id','cat1','cat2','cat3','cat4']

data1 = pd.DataFrame(data1)
#data2 = pd.DataFrame(data2)
#data2.columns = mycol
#data_merge = pd.merge(dadjta2, data1, on=("recipe_id"))


data_source = data1.loc[:,['rec_source','rec_step']]

#print(data_source.loc[data_source[1],'rec_source'])
recss = ['rec_source', 'rec_step']
#print(data_source[['rec_source', 'rec_step']])

#print(data_source)

rec_souce = []
rec_step = []
rec_step_adj = []

for idx in recss:
    for line in data_source[str(idx)]:
        if idx == 'rec_source':
            token_1 = okt.nouns(line)
            rec_souce.append(token_1)
        elif idx == 'rec_step':
            #print(line)
            token_2 = okt.nouns(line)
            rec_step.append(token_2)

            token_3 = okt.pos(line)
            a = []
            for token in token_3:
                if token[1] == 'Adjective':
                    b = token[0]
                    a.append(b)
            rec_step_adj.append(a)
            #news_word = okt.nouns(line)
            #print(news_word)

#print(rec_souce)
#print(rec_step)
#print(rec_step_adj)

data_source.loc[:,'rec_source'] = rec_souce
data_source.loc[:,'rec_step'] = rec_step
data_source['rec_step_pos'] = rec_step_adj

#print(data_source)
#[[0 for j in range(2)] for i in range(3)]

list1 = rec_souce
list2 = rec_step
print(type(rec_souce))
print(type(rec_step))
print(type(rec_step_adj))
#print(set(list1) == set(list2))
########################################
# 재료 토큰, 순서 토큰 내부 비교
n = 0
tot = []
for i in range(0,len(list1)):
    not_subset_token = []
    for x in list1[i]:
        if x not in list2[i]:
            # print(x)
            not_subset_token.append(x)
    tot.append(not_subset_token)
for l in tot:
    print(l)
print(n)

#########################################


data_source.to_csv('test_token.csv')



print("--- %s seconds ---" % (time.time() - start_time))


#data_merge.to_csv('test_merge.csv')
#######################################
# okt = Okt()
# filename = './crawl_data/recipe_info_99999_n.csv'
# #data = open('./crawl_data/recipe_info_99999_n.csv', 'rt', encoding='UTF-8')
# data = pd.read_csv(filename, encoding='UTF-8', index_col=0).head(10)
#  #column =',recipe_id,rec_title,rec_sub,rec_source,rec_step,rec_tag'
# print(type(data))
# data = pd.DataFrame(data)
# data = data['rec_source']
#
# print(data)
#
# for idx in data:
#     news_word = []
#     news_word = okt.nouns(idx)
#     print(news_word)
#
# data.to_csv('test_konlpy.csv')
################################################