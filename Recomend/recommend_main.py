import os
import cv2
import pickle
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from refrigerator_recipe.computer_vision.cv_predict import Prediction

def calculate_distance(x,y):
    lens = len(x)
    distance = 0
    for i in range(lens):
        distance += (x[i] - y[i])**2
    return math.sqrt(distance)

def calculate_consine(x,y):
    lens = len(x)
    x_scale = 0
    y_scale = 0
    xy_dot = 0
    for i in range(lens):
        x_scale += x[i]**2
        y_scale += y[i]**2
        xy_dot += x[i]*y[i]

    return xy_dot/(math.sqrt(x_scale)*math.sqrt(y_scale))

def find_point(SOURCES):
    SOURCE_LEN = len(SOURCES)

    find_point_x = 0
    find_point_y = 0
    find_point_z = 0

    for source in SOURCES:
        find_point_x += sourceFrame.loc[source,'x']
        find_point_y += sourceFrame.loc[source,'y']
        find_point_z += sourceFrame.loc[source,'z']

    # find_point_x = find_point_x/SOURCE_LEN
    # find_point_y = find_point_y/SOURCE_LEN
    # find_point_z = find_point_z/SOURCE_LEN
    return find_point_x,find_point_y,find_point_z

if __name__ == '__main__':

    # ############# 이미지 판별 #############
    # model_path = '../../data/computer_vision_data/resnet_extraction_224__epoch40_85.h5'
    # label_path = '../../data/computer_vision_data/label_dict.txt'
    # prediction = Prediction(model_path,label_path)
    #
    # # load image and convert to array( input shape)
    # img = load_img('C:/Users/TJ/Desktop/연어.jpg')
    # img_array = np.array(img)
    # img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA) /255.0
    # img_array = img_array.reshape(1,224,224,3)
    # prediction.predict_label(img_array)


    ############# 판별 이미지로 레시피 검색 #############
    source_path = '../../data/nlp_data/source_embedding.csv'
    recipe_path = '../../data/nlp_data/recipe_embedding.csv'
    sourceFrame = pd.read_csv(source_path,index_col=3)
    recipeFrame = pd.read_csv(recipe_path)
    print(sourceFrame.shape,'\n',recipeFrame.shape)

    sources = ['버터']
    select_cat = ('볶음','일상',None,None)

    my_point = find_point(sources)

    filter1Data = []
    for idx,row in recipeFrame.iterrows():
        recipe_category = (row.cat1,row.cat2,row.cat3,row.cat4)
        category_check = 0
        select_length = 0
        for i in range(4):
            if select_cat[i] is not None:
                select_length +=1
                if select_cat[i] == recipe_category[i]:
                    category_check +=1

        # 카테고리에 해당되는지 여부
        source_check = 0
        if select_length == category_check :
            recipe_sources = row.kwd_source.split('|')
            for source in sources:
                if source in recipe_sources:
                    source_check += 1
            if source_check > 0 :
                filter1Data.append({ 'id':row.id,
                                     'title':row.title,
                                     'x':row.x,
                                     'y':row.y,
                                     'z':row.z})
        print(select_length,category_check,source_check)
    # for recipeFrame.iterrows()

    FilterFrame = pd.DataFrame(filter1Data)
    print(FilterFrame.shape)

    distance_list = []
    for idx,row in FilterFrame.iterrows():
        recipe_point = (row.x,row.y,row.z)
        distance = calculate_distance(my_point,recipe_point)
        distance_list.append((distance,row.id,row.title,recipe_point))

    distance_list = sorted(distance_list,key=lambda x : x[0])[:50]
    [print(i) for i in distance_list]

    ############################### DRAW GRAPH ##################################
    # mpl.rcParams['legend.fontsize'] = 10  # 그냥 오른쪽 위에 뜨는 글자크기 설정이다.
    # mpl.rc('font', family='Malgun Gothic')
    # fig = plt.figure(figsize=(20,20))  # 이건 꼭 입력해야한다.
    # ax = fig.gca(projection='3d')
    # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # z = recipeFrame['z']
    # x = recipeFrame['x']
    # y = recipeFrame['y']
    # ax.plot(x, y, z,'o', label='parametric curve')  # 위에서 정의한 x,y,z 가지고 그래프그린거다.
    # ax.legend()  # 오른쪽 위에 나오는 글자 코드다. 이거 없애면 글자 사라진다. 없애도 좋다.
    # for t,x,y,z in zip(recipeFrame['title'],x,y,z):
    #     ax.text(x,y,z,t)
    # plt.show()