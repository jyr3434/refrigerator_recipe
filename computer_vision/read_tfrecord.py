import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.datasets import mnist

from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Activation # 레이어 추가
from tensorflow.keras import activations,optimizers,metrics #케라스 자체로만 하면 최신 버전 사용 가능
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout

# Create a description of the features.
def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value='' ),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0 )
        # 'x': tf.io.FixedLenFeature([],tf.int64, default_value=0),
        # 'y': tf.io.FixedLenFeature([],tf.int64, default_value=0),
        # 'z': tf.io.FixedLenFeature([],tf.int64, default_value=0)
    }
    # Load one example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Turn your saved image string into an array
    parsed_features['image'] = tf.io.decode_raw(parsed_features['image'], out_type=tf.float32)
    image = tf.cast(parsed_features['image'], tf.float32) / 255.0
    image = tf.reshape(image, [1,224, 224, 3])

    label = tf.one_hot(parsed_features['label'],5)
    label = tf.reshape(label, [1,5])
    # parsed_features['image'] = tf.reshape(parsed_features['image'],shape=(224,224,3))
    # return {'image':parsed_features['image'],'label':parsed_features["label"],'x':parsed_features['x'],
    #         'y':parsed_features['y'],'z':parsed_features['z']}
    return image,label

if __name__ == '__main__':
    train_dataset = tf.data.TFRecordDataset('../../data/computer_vision_data/train.tfrecord',compression_type='GZIP')
    train_dataset = train_dataset.map(_parse_function)

    # print(raw_dataset)

    # for raw_record in raw_dataset.take(10):
    #     # raw_record['image'] = tf.reshape(raw_record['image'] ,shape=(224,224,3))
    #     print(raw_record[1])



    # setting graph

    # TFrecord (체크)
    # 진행전 라벨링 작업, 전처리
    # 분류 갯수 설정
    nb_classes = 5
    model = Sequential()

    # convolution layer
    # padding = 'valid' (행열수 줄어듬), 'same' ( 행열수 보존 )
    # filter 개수는 보통 32,64...
    # Conv2D : input_shape -> ( 높이,너비,채널수 )
    # 채널수는 흑백 : 1  컬럼 : 3(RGB)
    # strides = (1,1) default
    # convolution도 여러개로
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=activations.relu, input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu))

    # 4차원 데이터를 2차원으로 축소하기
    model.add(Flatten())

    # full connected
    model.add(Dense(units=512, activation=activations.relu))
    model.add(Dropout(0.3))
    model.add(Dense(units=512, activation=activations.relu))
    model.add(Dropout(0.3))
    model.add(Dense(units=nb_classes, activation=activations.softmax))

    model.summary()

    model.compile(loss=losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

    print('fitting 중입니다.')  # checking workflow output
    with tf.device('/gpu:0'):
        model.fit(train_dataset, epochs=5, batch_size=1, verbose=0)

    print(model.metrics_names)

    ###########################################
    test_dataset = tf.data.TFRecordDataset('../../data/computer_vision_data/test.tfrecord',compression_type='GZIP')
    test_dataset = test_dataset.map(_parse_function)
    print('evaluate 중입니다.')  # checking workflow output
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    # model.metric_names와 model.evaluate의 return 결과물은 연관성이 있으니
    # 둘의 관계를 공부하자
    print('test_acc : %.4f' % test_acc)
    print('test_loss : %.4f' % test_loss)
    print('-' * 50)
    model.save('raw_data_model.h5')

