import re,time,urllib
import urllib.request
import urllib.error
import pandas as pd
from itertools import count
from bs4 import BeautifulSoup
from bs4 import element
from multiprocessing import Pool

class Crawl:
    # 초기 url 여기에 cat id를 합친다
    main_url = 'https://www.10000recipe.com/recipe/list.html'
    base_url = 'https://www.10000recipe.com/recipe/list.html?cat1={}&cat2={}&cat3={}'
    cat4_url = 'https://www.10000recipe.com/recipe/list.html?cat4={}'
    def __init__(self):
        # 대분류별(cat1~4) 중분류 정보를 담고 있는 변수 key : id_text / value : id
        self.cat1,self.cat2,self.cat3,self.cat4 = {},{},{},{}
        

    def GetCRL(self): # 대분류별 중분류 정보 수집
        req = urllib.request.Request(self.main_url)
        sourcecode = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(sourcecode, "html.parser")

        pattern = re.compile('[0-9]+')

        caturl = soup.find("div", id="id_search_category").find_all("div", class_="cate_list")
        result = []
        for allca in caturl:
            catall = allca.find_all('a')


            cat_dict = {}
            for a in catall[1:]:
                #print(a['href'])
                re_str = a['href'].split(',')[1]
                cat_id = pattern.search(re_str).group()
                cat_text = a.text
                cat_dict[cat_id] = cat_text

            result.append(cat_dict)
        self.cat4, self.cat2, self.cat3, self.cat1 =  result

    def Crawl_recipe_id(self,c1): # 중분류별 레시피 id 수집
        '''
        catergory1 & category2 & category3 을 동시에 만족하는 recipe들을 찾아 id와 그에 맞는 검색 기준을 attribute취급하여 같이 저장한다.

        count를 이용하여 board page를 넘기기 검색결과가 없으면 해당 category 수집 중단

        :param c1: 방법 분류 id
        :return: recipe_list ( 1 recipe :: 1 dict ) recipe id 가 어떤 중분류(방법, 상황, 재료) 정보를 담고 있는지 같이 저장된다.
        '''

        recipe_list = []
        for c2 in self.cat2.keys():
            for c3 in self.cat3.keys():
                for page_idx in count(1):
                    pageurl = self.base_url.format(c1,c2,c3) + '&page=' + str(page_idx) # 방법, 상황, 재료
                    print(pageurl)
                    sourcecode = urllib.request.urlopen(pageurl).read()
                    soup = BeautifulSoup(sourcecode, "html.parser")
                    # print(pageurl)
                    col_xs_list = soup.select("div.col-xs-3")  # 해당 상위 속성 범위 좁히기

                    if len(col_xs_list)>1 : # 검색 결과가 있습니다.
                        for col_xs in col_xs_list:
                            print('-'*60)

                            a = col_xs.select_one('a.thumbnail') # 광고/레시피 중 레시피 요소만 추출
                            if a:
                                recipe_id = a['href'].split('/')[2] # /recipe/id 중 id만 추출
                                print(recipe_id)
                                recipe_list.append({'id':recipe_id,\
                                                    'cat1':self.cat1[c1],\
                                                    'cat2':self.cat2[c2],\
                                                    'cat3':self.cat3[c3]})
                    else: # 검색결과가 없습니다 작업 중단
                        break
        # end for
        return recipe_list # 1페이지당 레시피의 결과물

    def Crawl_recipe_id_by_cat4(self,c4):
        '''
        category4를 이용하여 각 조건에 해당하는 레시피의 id를 추출한다.

        :param c4: 종류 분류 id
        :return: recipe_list ( 1 recipe :: 1 dict ) recipe id 가 어떤 중분류(종류) 정보를 담고 있는지 같이 저장된다.
        '''
        recipe_list = []

        for page_idx in count(1):
            pageurl = self.cat4_url.format(c4) + '&page=' + str(page_idx) # 종류
            print(pageurl)
            sourcecode = urllib.request.urlopen(pageurl).read()
            soup = BeautifulSoup(sourcecode, "html.parser")
            # print(pageurl)
            col_xs_list = soup.select("div.col-xs-3")  # 해당 상위 속성 범위 좁히기

            if len(col_xs_list)>1 : # 검색 결과가 있습니다.
                for col_xs in col_xs_list:
                    print('-'*60)
                    a = col_xs.select_one('a.thumbnail')
                    if a:
                        recipe_id = a['href'].split('/')[2]
                        print(recipe_id)
                        recipe_list.append({'id':recipe_id,\
                                            'cat4':self.cat4[c4]
                                            })
            else: # 검색결과가 없습니다
                break
        # end for
        return recipe_list # 1페이지당 레시피의 결과물

    def PageCrawler(self,recipe_id):
        rec_dict = {}
        Curl = 'https://www.10000recipe.com/recipe/'+ str(recipe_id)
        try:
            csourcecode = urllib.request.urlopen(Curl).read()
        except urllib.error.URLError:
            return rec_dict
        soup = BeautifulSoup(csourcecode, "html.parser")

        rec_title = []  # 레시피 제목
        rec_source = {}  # 레시피 재료
        rec_step = []  # 레시피 순서
        rec_dict['id'] = str(recipe_id)
        try:
            res = soup.find('div', 'view2_summary')
            res = soup.find('h3')
            rec_dict['rec_title']=res.get_text()

        except(AttributeError):
            rec_dict['rec_title']='-'

        try:
            res = soup.find('div', 'view2_summary_info')
            rec_dict['rec_sub']= res.get_text().replace('\n', '')
            res = soup.find('div', 'ready_ingre3')
        except(AttributeError):
            rec_dict['rec_sub']='-'

        rg = re.compile('[\s]{2,}')
        try:
            source_category = []
            for n in res.find_all('ul'):
                source_element_list = []
                title = n.find('b').get_text()
                rec_source[title] = ''
                for tmp in n.find_all('li'):
                    text = tmp.get_text().replace('\n', '')
                    text = rg.sub(' ',text)
                    source_element_list.append(text)
                source_category.append('|'.join(source_element_list))
            rec_dict['rec_source']='&'.join(source_category)
        except (AttributeError):
            rec_dict['rec_source']='-'

            #  요리 순서 찾는 for 문
        res = soup.find('div', 'view_step')
        i = 0
        if (res):
            for n in res.find_all('div', 'view_step_cont'):
                i = i + 1
                rec_step.append(n.get_text().replace('\n', ' '))
                #  나중 순서를 구별 하기 위해 #을 넣는다
            if rec_step:
                rec_dict['rec_step']='|'.join(rec_step)
            else:
                rec_dict['rec_step']='-'

            tag = res.find('div', 'view_tag')
            if tag:
                rec_dict['rec_tag'] = tag.get_text()
                # del recipe_tag[0]
            else:
                rec_dict['rec_tag'] = '-'
        else:
            rec_dict['rec_step'] = '-'
            rec_dict['rec_tag'] = '-'
        # 해시 태그가 글 내에 있는 판단하고 출력 해주는  for문


        print(rec_dict)

        return rec_dict

if __name__=='__main__':
    crawl = Crawl()
    # crawl.GetCRL()
    # print(crawl.cat4, crawl.cat2, crawl.cat3, crawl.cat1, sep='\n')
    print('크롤링 시작')
    start_time = time.time()

    df = pd.read_csv('../pre-processing/crawl_data/id_4category.csv',index_col=0)
    print(df.iloc[:5,0])
    pool = Pool(processes=16) # 4개의 프로세스를 사용합니다
    result = pool.map(crawl.PageCrawler, iter(df.iloc[70000:,0]))
    df = pd.DataFrame(result)
    '''
    # 중복되는 파일명으로 내용이 덮어쓰기가 될수 있으므로
    # file_serial_name으로 구별 시킨다.
    '''
    file_serial_name = 'serial_n'
    df.to_csv('../pre-processing/crawl_data/recipe_info_{}.csv'.format(file_serial_name))
    # result_list = pool.map(crawl.Crawl_recipe_id,iter(crawl.cat1.keys()))
    # end = []
    # [end.extend(i) for i in result_list]
    # df = pd.DataFrame(end)
    # df.to_csv('crawl_data/crawl_recipe_id.csv', encoding='UTF-8', header=True)

    # result_list = pool.map(crawl.Crawl_recipe_id_by_cat4,iter(crawl.cat4.keys()))
    # end = []
    # [end.extend(i) for i in result_list]
    # df = pd.DataFrame(end)
    # df.to_csv('crawl_data/cat4.csv', encoding='UTF-8', header=True)

    # df1 = pd.read_csv('crawl_data/crawl_recipe_id.csv',index_col=0)
    # df2 = pd.read_csv('crawl_data/cat4.csv',index_col=0)
    # df3 = pd.merge(df1,df2,on='id')
    # df3.to_csv('crawl_data/id_4category.csv',encoding='UTF-8',header=True)
    print("--- %s seconds ---" % (time.time() - start_time))