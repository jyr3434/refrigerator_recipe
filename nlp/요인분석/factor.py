from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab
import numpy as np
import pandas as pd
model = Word2Vec.load('../../../data/nlp_data/source_word2vec.model')
k = list(model.wv.vocab.keys())
pd.DataFrame([model[i] for i in k]).to_csv('../../../data/nlp_data/factor.csv',index=False)
