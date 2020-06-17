import tensorflow as tf
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    raw_dataset = tf.data.TFRecordDataset('../../data/computer_vision_data/test.tfrecord',compression_type='GZIP')
    raw_dataset = raw_dataset.map(_parse_function)

    # print(raw_dataset)
    i = 0
    for raw_record in raw_dataset.take(15000):
        i+=1
        print(i)
        # plt.figure()
        raw_record['image'] = tf.reshape(raw_record['image'],shape=(224,224,3))
        img = tf.keras.preprocessing.image.array_to_img(raw_record['image'])
        # plt.imshow(img)
        # plt.show()
        print(raw_record['image'])
