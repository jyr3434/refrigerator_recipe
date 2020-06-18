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

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


