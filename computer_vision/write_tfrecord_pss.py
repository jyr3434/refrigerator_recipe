import os
import random
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array


def get_path(folder):
    img_list = [(i[0], i[2]) for i in list(os.walk(f'../../data/crl_image/{folder}'))[1:]]
    print(len(img_list))
    return img_list  # [ ( str, list ) , ... ]

def seperate_data(img_path):
    train = []
    test = []

    for path,file_list in img_path:
        lens = len(file_list)
        random.seed(1000)
        random.shuffle(file_list)
        point = int(0.8*lens)
        train.append((path,file_list[:point]))
        test.append((path,file_list[point:]))
    return train,test

def label_dict(img_path):
    # 문자 라벨 숫자로 변환 144
    labeling_dict = {j[1]: j[0] for j in enumerate([i[0].split('\\')[-1] for i in img_path])}
    return labeling_dict

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
            _binary_image = image_ary.tostring()

            string_set = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(_binary_image),
                'label': _int64_feature(label)
            }))

            writer.write(string_set.SerializeToString())
    writer.close()

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

def suffle_data(data):
    shuffle_result = []
    for l,img_list in data:
        key = l.split('\\')[-1]
        imsi = [(img,key) for img in img_list]
        shuffle_result.extend(imsi)
    # print(len(shuffle_result))
    random.seed(54321)
    random.shuffle(shuffle_result)
    return shuffle_result

def shuffle_to_record(data,labeling_dict,groupfolder, tfrecords_name):
    print("Start converting")
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    writer = tf.io.TFRecordWriter(path=tfrecords_name, options=options)
    dirpath = f'../../data/crl_image/{groupfolder}'
    end = len(data)
    cnt = 0
    for filename,labelkey in data:
        label = labeling_dict[labelkey]
        filepath = '\\'.join((dirpath,labelkey,filename))
        image = load_img(filepath)
        image_ary = img_to_array(image)
        _binary_image = image_ary.tostring()

        string_set = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(_binary_image),
            'label': _int64_feature(label)
        }))
        writer.write(string_set.SerializeToString())
        cnt +=1
        if cnt % 10000 == 0 :
            print(cnt,'/',end)
    writer.close()
if __name__ == '__main__':
    # image_path_list = get_path('crl_image_resize')
    # labeling_dict = label_dict(image_path_list)
    # with open('../../data/computer_vision_data/label_dict.txt', 'w', encoding='utf-8') as f:
    #     for k,v in labeling_dict.items():
    #         f.write(str(v)+':'+k+'\n')
    # train,test = seperate_data(image_path_list)
    #
    # tfrecord_version = '_resize'
    #
    # train_name = f'train{tfrecord_version}'
    # test_name = f'test{tfrecord_version}'
    # to_tfrecords(train, labeling_dict, f'../../data/computer_vision_data/{train_name}.tfrecord')
    # to_tfrecords(test, labeling_dict, f'../../data/computer_vision_data/{test_name}.tfrecord')
    pass