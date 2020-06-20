from tensorflow.keras import models, layers
from tensorflow.python.keras import losses
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras import optimizers, initializers, regularizers, metrics
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from refrigerator_recipe.computer_vision.cv_model import ResNet,Own
from refrigerator_recipe.computer_vision.cv_dataset import DataSet
from refrigerator_recipe.computer_vision.cv_keras_model import keras_resnet50
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)
    with tf.device('/GPU:0'):
        inputs = (224,224,3)
        outputs = 144
        dataset = DataSet(inputs,outputs)
        print(' ############모델 선택하기##################\n'
              ' Own : o \n'
              ' ResNet : r \n'
              ' Keras : k \n'
              ' .... : \n')
        command_key = input('키를 입력하세요( 대소문자 상관없음 ) : ').lower()

        modelname_detail = 'v3' # 같은 모델이라도 옵션이 다를수 있는 부가적인 이름을 추가해쥇요

        # choice model and built model graph of end compile
        if command_key in ('r','ㄱ'):
            model = ResNet(inputs,outputs)
            modelname = f'resnet_{modelname_detail}'
        elif command_key in ('o','ㅐ'):
            model = Own(inputs,outputs)
            modelname = f'own_{modelname_detail}'
        elif command_key in ('k','ㅏ'):
            model = keras_resnet50(outputs)
            modelname = f'keras_{modelname_detail}'

        # create or load model path
        model_path = f'../../data/computer_vision_data/{modelname}_model.h5'
        dataset_version = '_v3'

        # if exist model load and evaluate
        # or not exist create model and fit(train) and evaluate and save model
        if os.path.isfile(model_path): # exist model
            print("Yes. it is a file")
            model.load_weights(model_path)
            test_dataset = dataset.tfrecord_dataset(f'../../data/computer_vision_data/test{dataset_version}.tfrecord')
            print('evaluate 중입니다.')
            test_loss, test_acc, test_top_k = model.evaluate(test_dataset,batch_size=10, verbose=1)
            print('test_acc : %.4f' % test_acc)
            print('test_loss : %.4f' % test_loss)
            print('-' * 50)
        elif os.path.isdir(model_path):
            print("Yes. it is a directory")
        elif os.path.exists(model_path):
            print("Something exist")
        else: # not exist model
            print("Nothing")
            train_dataset = dataset.tfrecord_dataset(f'../../data/computer_vision_data/train{dataset_version}.tfrecord')
            print('fitting 중입니다.')
            model.fit(train_dataset, epochs=1, batch_size=30, verbose=1)

            test_dataset = dataset.tfrecord_dataset(f'../../data/computer_vision_data/test{dataset_version}.tfrecord')
            print('evaluate 중입니다.')
            test_loss, test_acc, test_top_k = model.evaluate(test_dataset, verbose=1)
            print('test_acc : %.4f' % test_acc)
            print('test_loss : %.4f' % test_loss)
            print('test_top_k : %.4f' % test_top_k)
            print('-' * 50)
            model.save(model_path)
