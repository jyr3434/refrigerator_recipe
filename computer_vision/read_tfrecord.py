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
    image = tf.reshape(image, [224, 224, 3])

    label = tf.one_hot(parsed_features['label'],144)
    # parsed_features['image'] = tf.reshape(parsed_features['image'],shape=(224,224,3))
    # return {'image':parsed_features['image'],'label':parsed_features["label"],'x':parsed_features['x'],
    #         'y':parsed_features['y'],'z':parsed_features['z']}
    return image,label

if __name__ == '__main__':
    raw_dataset = tf.data.TFRecordDataset('../../data/computer_vision_data/test.tfrecord',compression_type='GZIP')
    raw_dataset = raw_dataset.map(_parse_function)

    # print(raw_dataset)

    for raw_record in raw_dataset.take(10):
        # raw_record['image'] = tf.reshape(raw_record['image'] ,shape=(224,224,3))
        print(raw_record[1])

