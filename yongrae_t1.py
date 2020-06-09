import os
import re
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import time
from multiprocessing import Pool

mainurl = 'https://www.10000recipe.com/recipe/list.html?order=reco&page=1'
# def get_count(num, p=4):
#     list = []
#     allocate = int(num/p)
#     for n in range(p):
#         list.append(allocate)
#     list[p-1] += num%p
#     print("프로세스 할당량", list)
#     return list

def PageCrawler(urllist):
    Curl = urllist
    creq = urllib.request.Request(Curl)
    csourcecode = urllib.request.urlopen(creq).read()
    soup = BeautifulSoup(csourcecode, "html.parser")


    rec_title = []  # 레시피 제목
    rec_source = []  # 레시피 재료
    rec_step = []  # 레시피 순서
    rec_tag = []  # 레시피 해시태그
    try :
        res = soup.find('div', 'view2_summary')
        res = soup.find('h3')
        rec_title.append(res.get_text())
        res = soup.find('div', 'view2_summary_info')
        rec_title.append(res.get_text().replace('\n', ''))

        res = soup.find('div', 'ready_ingre3')
    except(AttributeError):
        pass

    pattern = re.compile('[\s]{2,}')
    try:
        for n in res.find_all('ul'):
            source = []
            title = n.find('b').get_text()
            #rec_source[title] = ''
            for tmp in n.find_all('li'):
                t = tmp.get_text().replace('\n', '')
                t = pattern.sub(' ',t)
                #source.append(t)
            #rec_source[title] = source
                source = title + t
    except (AttributeError):
        return

        #  요리 순서 찾는 for 문
    res = soup.find('div', 'view_step')
    i = 0
    rec_step_imsi = []
    for n in res.find_all('div', 'view_step_cont'):
        i = i + 1
        rec_step_imsi.append(n.get_text().replace('\n', ' '))
    rec_step.append('|'.join(rec_step_imsi))

    # 해시 태그가 글 내에 있는 판단하고 출력 해주는  for문
    if (res.find('div', 'view_tag')):
        rec_tag = res.find('div', 'view_tag').get_text()
        # del recipe_tag[0]

    # 블로그 형식의 글은 스텝이 되어있지 않기 때문에 제외 해준다
    if not rec_step:
        return

    recipe_all = [rec_title, source, rec_step, rec_tag]
    return (recipe_all)

def Getrcp(page_idx):

    reclist = []


    pageurl = mainurl + '&page=' + str(page_idx)
    req = urllib.request.Request(pageurl)
    sourcecode = urllib.request.urlopen(pageurl).read()
    soup = BeautifulSoup(sourcecode, "html.parser")
    print(pageurl)
    allrcp = soup.find("div",class_="row").find_all("div",class_="col-xs-3") # 해당 상위 속성 범위 좁히기
    '''
    #print(allrsp.find("a")["href"])
    #print(type(allrcp))
    #print('-'*60)
    #print(allrsp)
    '''
    baseurl = 'https://www.10000recipe.com/'
    #
    for idx in allrcp:
        print('-'*60)
        if idx.find('a') == None: # A태그에 해당안되어 None 값으로 리턴되는 결과 제외
            pass
        else:
            rcpurl = idx.find('a')['href']
            #print(rcpurl)
            full_url = baseurl + rcpurl
            print(full_url)
            # url에서 원하는 데이터 추출하기
            reclist.append(PageCrawler(full_url))
    print(len(reclist))
            # i = i + 1
            # print("count:",i)
    return reclist # 1페이지당 레시피의 결과물





if __name__=='__main__':
    print('크롤링 시작')
    urllist = []
    start_time = time.time()
    pool = Pool(processes=16) # 4개의 프로세스를 사용합니다

    resultlist = pool.map(Getrcp,range(1,2)) # get_contetn 함수를 넣어줍시다.
    end = []
    [end.extend(i) for i in resultlist]
    df = pd.DataFrame(end)
    df.to_csv('allrecdf.csv', encoding='UTF-8', header=False)
    print("--- %s seconds ---" % (time.time() - start_time))


#Getrcp()
# print(len(urllist))
# print(urllist)


# print(urllist[0])
# print(reclist)

# recdf = pd.DataFrame()
# recdf.to_csv('allrecdf.csv', encoding='UTF-8', header=False)