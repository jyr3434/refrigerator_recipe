from refrigerator_recipe.computer_vision.cv_dataset import DataSet
from refrigerator_recipe.computer_vision.write_tfrecord import get_path,to_tfrecords,label_dict,seperate_data
import tensorflow as tf
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
# matplotlib.matplotlib_fname()

#rc['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import time
import random
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array

#import font to use at the outcome window
#rc('font', familly='AppleGothic')
def x_y(data,labeling_dict):
    x_l = []
    y_l = []
    for dirpath, image_list in data:
        labelkey = dirpath.split('\\')[-1]
        label = labeling_dict[labelkey]
        print(labelkey)
        for image_path in image_list:
            filepath = '\\'.join((dirpath, image_path))

            image = load_img(filepath)
            image_ary = img_to_array(image)
            label_ary = tf.one_hot(label,144)
            x_l.append(image_ary)
            y_l.append(label_ary)
    return np.array(x_l),np.array(y_l)

if __name__ == '__main__':
    # img_lsit = get_path('crl_image_extraction_64')
    # train, test = seperate_data(img_lsit)
    # labeling_dict = label_dict(test)
    #
    # x_train, y_train = x_y(train[0:10], labeling_dict)
    # del train
    # x_test, y_test = x_y(test[0:10], labeling_dict)
    # del test
    # train_num = x_train.shape[0]
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)
    with tf.device('/GPU:0'):
        learning_rate = 0.001
        batch_size = 500
        training_epochs = 500

        # print(x_train.shape)
        #
        # print(y_train.shape)


        @tf.function
        def cnn_layers(x_img):
            w1 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
            b1 = tf.Variable(tf.constant(0.1, shape=[64]))
            w2 = tf.Variable(tf.random.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))
            b2 = tf.Variable(tf.constant(0.1, shape=[64]))
            w3 = tf.Variable(tf.random.normal(shape=[3, 3, 64, 128], stddev=5e-2))
            b3 = tf.Variable(tf.constant(0.1, shape=[128]))
            w4 = tf.Variable(tf.random.normal(shape=[3, 3, 128, 128], stddev=5e-2))
            b4 = tf.Variable(tf.constant(0.1, shape=[128]))
            w5 = tf.Variable(tf.random.normal(shape=[3, 3, 128, 128], stddev=5e-2))
            b5 = tf.Variable(tf.constant(0.1, shape=[128]))

            L1 = tf.nn.relu(tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, 0.5)

            L2 = tf.nn.relu(tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, 0.5)

            L3 = tf.nn.relu(tf.nn.conv2d(L2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
            L3 = tf.nn.dropout(L3, 0.5)

            L4 = tf.nn.relu(tf.nn.conv2d(L3, w4, strides=[1, 1, 1, 1], padding='SAME') + b4)
            L4 = tf.nn.dropout(L4, 0.5)

            L5 = tf.nn.relu(tf.nn.conv2d(L4, w5, strides=[1, 1, 1, 1], padding='SAME') + b5)
            L5 = tf.nn.dropout(L5, 0.5)
            L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # fully_connected1
            L5_flat = tf.reshape(L5, [-1, 128 * 8 * 8])

            initial = tf.initializers.GlorotNormal()
            fc_w1 = tf.Variable("w4",initial(shape=(128 * 8 * 8, 384)))
            fc_b1 = tf.Variable(tf.random.normal([384]))
            fc_l1 = tf.nn.relu(tf.matmul(L5_flat, fc_w1) + fc_b1)
            fc_l1 = tf.nn.dropout(fc_l1, 0.5)

            initial = tf.initializers.GlorotNormal()
            fc_w2 = tf.Variable("w5",initial(shape=(384, 144)))
            fc_b2 = tf.Variable(tf.random.normal([144]))
            logits = tf.matmul(fc_l1, fc_w2) + fc_b2

            y_pred = tf.nn.softmax(logits)
            return y_pred
            # print(y_pred.get_shape())
        ##########################################################
        # define loss and optimizer
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adadelta()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = cnn_layers(images)
                loss = loss_object(y_true=labels, y_pred=predictions)
            gradients = tape.gradient(loss, cnn_layers.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, cnn_layers.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def valid_step(images, labels):
            predictions = cnn_layers(images, training=False)
            v_loss = loss_object(labels, predictions)

            valid_loss(v_loss)
            valid_accuracy(labels, predictions)

        # start training
        Epochs = 10
        batch_size = 32
        inputs = (64, 64, 3)
        outputs = 144
        dataset = DataSet(inputs, outputs)
        dataset_version = '_extraction_64'

        train = dataset.tfrecord_dataset(f'../../data/computer_vision_data/train{dataset_version}.tfrecord')
        train = train.shuffle(buffer_size=500)
        test = dataset.tfrecord_dataset(f'../../data/computer_vision_data/test{dataset_version}.tfrecord')
        test = test.shuffle(buffer_size=500)

        for epoch in range(Epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            step = 0
            for images, labels in train:
                step += 1
                train_step(images, labels)
                print("Epoch: {}/{}, , loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1, Epochs,train_loss.result(),train_accuracy.result()))

            for valid_images, valid_labels in test:
                valid_step(valid_images, valid_labels)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                  "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                      Epochs,
                                                                      train_loss.result(),
                                                                      train_accuracy.result(),
                                                                      valid_loss.result(),
                                                                      valid_accuracy.result()))

        # cnn_layers.save_weights(filepath='../../data/computer_vision_data/cv_nn_model.h5', save_format='tf')



