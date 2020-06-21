import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array


def get_path(folder):
    img_list = [(i[0], i[2]) for i in list(os.walk(f'../../data/crl_image/{folder}'))[1:]]
    print(len(img_list))
    # for i in img_list:
    #     print(i[0],i[1],sep='\n')
    return img_list  # [ ( str, list ) , ... ]

def seperate_data(img_path):
    train = []
    test = []

    for path,file_list in img_path:
        lens = len(file_list)
        random.seed(1000)
        random.shuffle(file_list)
        point = int(0.7*lens)
        train.append((path,file_list[:point]))
        test.append((path,file_list[point:]))
    return train,test

def label_dict(img_path):
    # 문자 라벨 숫자로 변환 144
    labeling_dict = {j[1]: j[0] for j in enumerate([i[0].split('\\')[-1] for i in img_path])}
    return labeling_dict

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""
    if isinstance(text, str):
        return text
    elif isinstance(text, 'unicode'):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)


def to_tfrecords(data,labeling_dict, tfrecords_name):
    print("Start converting")
    options = tf.io.TFRecordOptions(compression_type = 'GZIP')
    writer = tf.io.TFRecordWriter(path=tfrecords_name, options=options)

    for dirpath,image_list in data:
        labelkey = dirpath.split('\\')[-1]
        label = labeling_dict[labelkey]
        print(labelkey)
        for image_path in image_list:
            filepath = '\\'.join((dirpath,image_path))

            image = load_img(filepath)
            image_ary = img_to_array(image)
            # print(image_ary)
            # print(image_ary.shape)
            _binary_image = image_ary.tostring()

            # print(repr(_binary_image))
            # _binary_label = labeling_dict[label].tobytes()
            # filename = os.path.basename(filepath)

            string_set = tf.train.Example(features=tf.train.Features(feature={
                # 'x': _int64_feature(image_ary.shape[0]),
                # 'y': _int64_feature(image_ary.shape[1]),
                # 'z': _int64_feature(image_ary.shape[2]),
                'image': _bytes_feature(_binary_image),
                'label': _int64_feature(label)
                # 'mean': _float_feature(image.mean().astype(np.float32)),
                # 'std': _float_feature(image.std().astype(np.float32)),
                # 'filename': _bytes_feature(str.encode(filename))
            }))

            writer.write(string_set.SerializeToString())
    writer.close()

if __name__ == '__main__':
    image_path_list = get_path('crl_image_resize')
    labeling_dict = label_dict(image_path_list)
    with open('../../data/computer_vision_data/label_dict.txt', 'w', encoding='utf-8') as f:
        for k,v in labeling_dict.items():
            f.write(str(v)+':'+k+'\n')
    train,test = seperate_data(image_path_list)

    point = 10
    # train_name = f'train_{point}'
    # test_name = f'test_{point}'
    tfrecord_version = '_resize'


    train_name = f'train{tfrecord_version}'
    test_name = f'test{tfrecord_version}'
    to_tfrecords(train, labeling_dict, f'../../data/computer_vision_data/{train_name}.tfrecord')
    to_tfrecords(test, labeling_dict, f'../../data/computer_vision_data/{test_name}.tfrecord')