import numpy as np
import pandas as pd
import cx_Oracle
import re
from multiprocessing import Pool


class RecipePreProcess:
    def __init__(self):
        pass
    def concat_data(self,x,y):
        return pd.concat([x,y],axis=0,ignore_index=True)
    def merge_data(self,x,y,on):
        return pd.merge(x,y,how='inner',left_on=on,right_on=on)

    # 파일명과 데이터 프레임을 넣어 준다.
    def save_data(self,filename,df):
        if isinstance(df,pd.DataFrame):
            # 파일명, 데이터의 범위를 표시해준다.
            df.to_csv('crawl_data/{}_0_{}.csv'.format(filename,len(df)),encoding='utf-8')
        else:
            print('argument type not dataframe')

    def str_drop_na(self,df):
        print('drop before',df.shape)
        for x in df.columns:
            df.loc[df[x] == '-'] = np.nan
        df = df.dropna(axis=0) #(125964, 10)

        for x in range(0,9):
            print(sum(df.iloc[:,x] == '-'))
        print('drop after',df.shape)
        return 0

    def match_db_df(self,df):
        conn,cur = None,None
        rg = re.compile('^[0-9]+$')
        try:
            loginfo = 'recommend/oracle@localhost:1521/xe'
            conn = cx_Oracle.connect(loginfo, encoding='utf-8')
            cur = conn.cursor()

            db_set = set()
            sql = " select id from recipe_infos "
            lists = []
            for i in cur.execute(sql):

                if i[0] and rg.match(i[0]):
                    db_set.add(int(i[0]))

            print(lists)
            df_set = set(df['id'])

            intersection_id = df_set - db_set
            return  df.loc[intersection_id,:]
        except Exception as err:
            print(err)
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    # 해당 데이터프레임을 oracle table에 insert한다.
    '''
    (1383, id                                                      6930519
    cat1                                                         볶음
    cat2                                                         일상
    cat3                                                       돼지고기
    cat4                                                       메인반찬
    rec_title                                               [돼지불고기]
    rec_sub                                         3인분 30분 이내 아무나 
    rec_source    돼지고기 앞다리살 760g|양파 1/2개|쪽파 1줌|다진마늘 1T|고추장 3T|설탕...
    rec_step      재료와 양념은 사진상으로 참고하시면 좋을 것 같아서 올려봅니다 ^^|돼지고기 앞다리...
    '''
    def df_to_oracle(self,df): # insert data to oracle table
        conn = None
        cur = None
        try:
            loginfo = 'recommend/oracle@localhost:1521/xe'
            conn = cx_Oracle.connect(loginfo, encoding = 'utf-8')
            cur = conn.cursor()
            if isinstance(df,pd.DataFrame):
                print(max([ len(i) for i in df['rec_step']]))
                for idx,row in df.iterrows():
                    sql = f" insert into recipe_infos " \
                          f"(id, cat1, cat2, cat3, cat4) " \
                          f" values('{row.id}','{row.cat1}','{row.cat2}','{row.cat3}'," \
                          f" '{row.cat4}')"
                    cur.execute(sql)
                    sql = " update recipe_infos set "
                    sql += self.execute_to_clob('rec_title',row.rec_title)+" , "
                    sql += self.execute_to_clob('rec_sub',row.rec_sub)+" , "
                    sql += self.execute_to_clob('rec_source',row.rec_source)+" , "
                    sql += self.execute_to_clob('rec_step',row.rec_step)
                    sql += f" where id = '{row.id}'"
                    cur.execute(sql)
                    print(idx)
                    pd.re
                conn.commit()
            else:
                print('argument type not dataframe')
        except Exception as err:
            print(err)
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    def execute_to_clob(self,col,text):
        sql = f" {col} = "
        lens = len(text)
        text = text.replace("'",'"')
        to_clob_list = []
        for x in range(0,lens,1000):
            if lens - x < 1000:
                to_clob_list.append(f" to_clob(' {text[x:]} ')")
            else:
                to_clob_list.append(f" to_clob(' {text[x:1000+x]} ')")
        sql = sql + " || ".join(to_clob_list)

        # print(sql)
        return sql


if __name__ == '__main__':
    recipepp = RecipePreProcess()
    #############
    # df_to_oracle
    df = pd.read_csv('../../data/crawl_data/recipe_data_dropna.csv')
    print(df.shape)
    print(recipepp.match_db_df(df).shape)
    # recipepp.df_to_oracle(df)
    ###################


    ################# multi process ######################
    # loginfo = 'recommend/oracle@localhost:1521/xe'
    # pool = cx_Oracle.SessionPool(loginfo,min=2,max=5,increment = 1,threaded=True,encoding='utf-8')
    # conn = pool.acquire()
    #
    # recipepp.conn.commit()
    # recipepp.conn.close()
    ##########################################

    # df1 = pd.read_csv('crawl_data/recipe_info_0_60000.csv',index_col=0)
    # df2 = pd.read_csv('crawl_data/recipe_info_60000_135345.csv',index_col=0)
    #
    # df3 = recipepp.concat_data(df1,df2)
    # cat4_df = pd.read_csv('crawl_data/id_4category.csv',index_col=0)
    # print(df3.shape,len(df3))
    # recipepp.save_data('recipe_info',df3)
    #
    # df4 = recipepp.merge_data(cat4_df,df3)
    # print(df4.shape,len(df4))
    # recipepp.save_data('recipe_data_final',df4)

    # df = pd.read_csv('crawl_data/recipe_data_final_0_135326.csv',index_col=0)
    # print(df.columns)
    '''
    Index(['id', 'cat1', 'cat2', 'cat3', 'cat4', 'recipe_id', 'rec_title',
       'rec_sub', 'rec_source', 'rec_step', 'rec_tag'],
      dtype='object')
    '''
    # for x in range(0,11):
    #     print(sum(df.iloc[:,x] == '-'))
    '''
    315
    307
    9747
    11057
    135318
    8
    '''
    # print(sum(df['rec_tag'] != '-'))
    # print(df.loc[df['rec_tag'] != '-','rec_tag'])
    # # 결측치 제거

    # df_dropna = df.drop(columns=['recipe_id','rec_tag'])
    #################################################################

    # df.to_csv('../../data/crawl_data/recipe_data_dropna.csv',encoding='utf-8',index=False)
    '''
    D:\Phycharm_pss\global_interpreter\venv\lib\site-packages\pandas\core\ops\array_ops.py:253:
    FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
    res_values = method(rvalues)
    '''