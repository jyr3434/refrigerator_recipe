import pandas as pd

df = pd.read_csv('../../data/crawl_data/recipe_data_dropna.csv')
s = df.style
df['id'] = [str(int(i)) for i in df['id']]
print(df['id'])
df.to_csv('../../data/crawl_data/recipe_data_dropna.csv',encoding='utf-8',index=False)