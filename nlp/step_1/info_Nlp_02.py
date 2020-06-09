from konlpy.tag import Okt,Mecab
from konlpy.corpus import kobill
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re,time

okt = Okt()
#mec = Mecab()


filename1 = '../../../data/crawl_data/recipe_data_dropna.csv'
data1 = pd.read_csv(filename1, encoding='UTF-8', index_col=0).head(10)
#mycol = ['recipe_id','cat1','cat2','cat3','cat4']
data1 = pd.DataFrame(data1)
data_source = data1.loc[:,['rec_source','rec_step']]
recss = ['rec_source', 'rec_step']

# 불용어 확보를 위한 테스트
for item1 in data_source.iloc[:,0]:
    #print(item)

    ang = re.findall('[0-9][^0-9]',item1)
    print(ang)




#print(type(data_source))
#print(data_source.iloc[:,0])


