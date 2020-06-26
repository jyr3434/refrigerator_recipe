import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Project03.refrigerator_recipe.computer_vision.cv_dataset import DataSet
from Project03.refrigerator_recipe.computer_vision.cv_model import ResNet
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
import glob


if __name__ == '__main__':
    # print(tf.one_hot(0,10))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    with tf.device('/GPU:0'):
        image_result = []
        folder = 'D:/pss/python/refrigerator_recommend/data/test_image/'
        model = ResNet((224, 224, 3), 144)
        print(model)
        model_path = '../../data/computer_vision_data/resnet_extraction_224__epoch40_85.h5'
        model.load_weights(model_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy', 'top_k_categorical_accuracy', 'categorical_crossentropy'])
        with open('../../data/computer_vision_data/label_dict.txt', 'r', encoding='utf-8') as f:
            labeling_dict = dict()
            label_list = f.readlines()
            for label in label_list:
                key, value = label.strip('\n').split(':')
                labeling_dict[key] = value
        for imgpath in glob.glob(folder + '*.jpg'):
            x = imgpath.split('\\')
            x.reverse()
            filename = x[0]
            # load image and convert to array( input shape)
            print('predict')
            img = load_img(str(imgpath))
            img_array = np.array(img)
            img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA) /255.0
            prediction = model.predict(img_array.reshape(1,224,224,3))
            # one_hot_encoding convert to integer
            answer_key = np.argmax(prediction[0])
            sf = tf.nn.softmax(prediction[0])
            # answer_key = np.where(prediction[0]==1)[0][0]
            answer = labeling_dict[str(int(answer_key))]
            image_result.append(filename+':'+answer)
        for item in image_result:
            print(item)
            # print(sf)


            # inputs = (224, 224, 3)
            # outputs = 10
            # epochs = 100
            # batchs = 4
            # opt = 'adam'
            # dataset = DataSet(inputs, outputs)
            # dataset_version = '_extraction_224_cat_10'
            # valid_dataset = dataset.tfrecord_dataset(f'../../data/computer_vision_data/valid{dataset_version}.tfrecord')
            # valid_dataset = valid_dataset.batch(batchs)
            #
            # model = ResNet(inputs,outputs)
            #
            # model.load_weights('../../data/computer_vision_data/resnet_extraction_224_cat_10_checkpoint.h5')
            # model.compile(loss='categorical_crossentropy', optimizer=opt,
            #               metrics=['accuracy', 'top_k_categorical_accuracy', 'categorical_crossentropy'])
            #
            # print('evaluate 중입니다.')
            # test_loss, test_acc, test_top_k, test_cate_cross = model.evaluate(valid_dataset, verbose=1)
            # print('test_acc : %.4f' % test_acc)
            # print('test_loss : %.4f' % test_loss)
            # print('test_top_k : %.4f' % test_top_k)
            # print('test_categoricat_crossentropy : %.4f' % test_cate_cross)
            # print('-' * 50)
