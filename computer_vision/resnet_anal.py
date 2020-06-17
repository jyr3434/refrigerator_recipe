from tensorflow.keras import applications
import tensorflow as tf
# resnet = applications.ResNet50(weights='imagenet')
# resnet.summary()
# json_string = resnet.to_json()


# with open('../../data/computer_vision_data/model.json','w') as f :
#     f.write(json_string)

import json
# parse = json.loads(json_string)
# print(json.dumps(parse, indent=4, sort_keys=True))

# pp_json_string = json.dumps(parse, indent=4, sort_keys=True)
# with open('../../data/computer_vision_data/model.json','w') as f :
#     f.write(pp_json_string)

# print(tf.keras.__version__)
# with tf.compat.v1.Session() as sess:
#   h = tf.constant("Hello")
#   w = tf.constant("World!")
#   hw = h + w
#   ans = sess.run(hw)
#   print(ans)

# [출처] 파이썬 텐서플로 AttributeError: module 'tensorflow' has no attribute 'Session' 문제 해결 방법|작성자 까미유
# 출처: https://mellowlee.tistory.com/entry/Windows-Tensorflow-Keras-사전-준비 [잠토의 잠망경]

# reference /Users/terrycho/dev/workspace/objectdetection/models/object_detection/data_decoders
# reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

# Create a description of the features.
def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value='' ),
        'label': tf.io.FixedLenFeature([], tf.string, default_value='' ),
        'x': tf.io.FixedLenFeature([],tf.int64, default_value=0),
        'y': tf.io.FixedLenFeature([],tf.int64, default_value=0),
        'z': tf.io.FixedLenFeature([],tf.int64, default_value=0)
    }
    # Load one example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Turn your saved image string into an array
    parsed_features['image'] = tf.io.decode_raw(
      parsed_features['image'], out_type=tf.float32)
    # print(parsed_features['image'].shape)
    return {'image':parsed_features['image'],'label':parsed_features["label"],'x':parsed_features['x'],
            'y':parsed_features['y'],'z':parsed_features['z']}

# string_set = tf.train.Example(features=tf.train.Features(feature={
#             'height': _int64_feature(image_ary.shape[0]),
#             'width': _int64_feature(image_ary.shape[1]),
#             'Image': _bytes_feature(_binary_image),
#             'Label': _bytes_feature(label.encode()),
#             # 'mean': _float_feature(image.mean().astype(np.float32)),
#             # 'std': _float_feature(image.std().astype(np.float32)),
#             'filename': _bytes_feature(str.encode(filename)),
#         }))

# create_dataset('../../data/computer_vision_data/imgtfrecord.tfrecord')

raw_dataset = tf.data.TFRecordDataset('../../data/computer_vision_data/test.tfrecord',compression_type='GZIP')
raw_dataset = raw_dataset.map(_parse_function)
# print(raw_dataset)
for raw_record in raw_dataset.take(10):
  print(raw_record['image'])
