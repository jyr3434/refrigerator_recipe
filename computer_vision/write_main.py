from refrigerator_recipe.computer_vision.write_tfrecord import get_path,to_tfrecords,label_dict

if __name__ == '__main__':

    #  test , train2
    train = get_path('train')
    test = get_path('test')
    labeling_dict = label_dict(test)
    tfrecord_version = '_v3'

    train_name = f'train{tfrecord_version}'
    test_name = f'test{tfrecord_version}'
    to_tfrecords(train[0:5], labeling_dict, f'../../data/computer_vision_data/{train_name}.tfrecord')
    to_tfrecords(test[0:5], labeling_dict, f'../../data/computer_vision_data/{test_name}.tfrecord')