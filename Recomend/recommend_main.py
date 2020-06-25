import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from refrigerator_recipe.computer_vision.cv_predict import Prediction

def find_point(SOURCES):
    SOURCE_LEN = len(SOURCES)

    find_point_x = 0
    find_point_y = 0
    find_point_z = 0

    for source in SOURCES:
        find_point_x += sourceFrame.loc[source,'x']
        find_point_y += sourceFrame.loc[source,'y']
        find_point_z += sourceFrame.loc[source,'z']

    find_point_x = find_point_x/SOURCE_LEN
    find_point_y = find_point_y/SOURCE_LEN
    find_point_z = find_point_z/SOURCE_LEN
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

    my_point = find_point(['청경채','고추','고기'])
    
    for idx,row in recipeFrame.iterrows():
        print(row.x,row.y,row.z)

