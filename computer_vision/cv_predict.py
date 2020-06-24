import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from refrigerator_recipe.computer_vision.cv_dataset import DataSet
from refrigerator_recipe.computer_vision.cv_model import ResNet
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array

if __name__ == '__main__':
    # print(tf.one_hot(0,10))
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    with tf.device('/GPU:0'):
        model = ResNet((224,224,3),144)
        print(model)
        model_path = '../../data/computer_vision_data/resnet_extraction_224__epoch30_79.hdf5'
        model.load_weights(model_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy', 'top_k_categorical_accuracy', 'categorical_crossentropy'])

        with open('../../data/computer_vision_data/label_dict.txt','r',encoding='utf-8') as f :
            labeling_dict = dict()
            label_list = f.readlines()
            for label in label_list:
                key,value = label.split(':')
                labeling_dict[key] = value

        # load image and convert to array( input shape)
        print('predict')

        print('model')
        img = load_img('C:/Users/brian/Desktop/버섯.jpg')
        print(img)
        img_array = np.array(img)
        img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA) /255.0
        print(img_array.shape)
        prediction = model.predict(img_array.reshape(1,224,224,3))
        print(prediction[0])
        # one_hot_encoding convert to integer
        answer_key = np.argmax(prediction[0])
        print(answer_key)
        sf = tf.nn.softmax(prediction[0])
        # answer_key = np.where(prediction[0]==1)[0][0]
        print(answer_key)
        print(labeling_dict)
        answer = labeling_dict[str(int(answer_key))]
        print(answer)
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
