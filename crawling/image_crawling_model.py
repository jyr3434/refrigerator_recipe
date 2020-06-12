from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from gensim.models import Word2Vec

model = Word2Vec.load('../../data/nlp_data/source_word2vec.model')
vocab = model.wv.vocab
print(len(vocab.keys()))

keydf = pd.read_csv('../../data/nlp_data/kwd_step_noun_okt.csv', index_col=0).head(5)
keydf.set_index(keys='id', inplace=True)
keydf_wd = keydf.iloc[:,0]

for item in keydf_wd:
    if isinstance(item,str):
        item  = item.split('|')
    # print(item)


# print(type(keydf_wd))
# print(keydf_wd)


def image_crl(self, keyword):
    search = str(keyword)

    url = f'https://www.google.com/search?q={quote_plus(search)}&source=lnms&tbm=isch&sa=X&ved=' \
          f'2ahUKEwjtkrie6fvpAhWCMN4KHXxsArQQ_AUoAXoECBYQAw&biw=1920&bih=920'

    path = 'D:\chromedriver.exe'
    driver = webdriver.Chrome(path)
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html)
    img = soup.select('.rg_i.Q4LuWd')
    n = 1
    imgurl = []

    for i in img:
        try:
            imgurl.append(i.attrs["src"])
        except KeyError:
            imgurl.append(i.attrs["data-src"])

    for i in imgurl:
        urlretrieve(i, "../../data/crl_image/" + search + str(n) +  ".jpg")
        n += 1
        print(imgurl)

    driver.close()


