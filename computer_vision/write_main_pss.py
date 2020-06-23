from refrigerator_recipe.computer_vision.write_tfrecord_pss import get_path,to_tfrecords,label_dict,seperate_data,suffle_data,shuffle_to_record
from refrigerator_recipe.pre_processing.image_generator import ImgGenerator
from multiprocessing import Pool

if __name__ == '__main__':

    # #  test , train2
    # train = get_path('train')
    # test = get_path('test')
    # labeling_dict = label_dict(test)
    # tfrecord_version = '_v3'
    #
    # train_name = f'train{tfrecord_version}'
    # test_name = f'test{tfrecord_version}'
    # to_tfrecords(train, labeling_dict, f'../../data/computer_vision_data/{train_name}.tfrecord')
    # to_tfrecords(test, labeling_dict, f'../../data/computer_vision_data/{test_name}.tfrecord')

<<<<<<< HEAD
    groupfolder = 'crl_image_resize_extraction_11'
=======
    groupfolder = 'crl_image_resize_extraction_10'
>>>>>>> 155828ac7203f4587a19cd57d15be0b78e641679
    IG = ImgGenerator()
    img_list = IG.get_path(groupfolder)
    # print(img_list[0])
    for fp, imgs in img_list:
        IG.generator(fp, imgs,groupfolder)

    img_list = get_path(groupfolder)
    # train, test = seperate_data(img_list)
    train,valid = seperate_data(img_list)
    labeling_dict = label_dict(valid)

<<<<<<< HEAD
    tfrecord_version = 'crl_image_11'
    train = suffle_data(train)
    valid = suffle_data(valid)
    test = suffle_data(test)
=======
    tfrecord_version = '_extraction_224_cat_10'
    train = suffle_data(train[0:10])
    valid = suffle_data(valid[0:10])
    # test = suffle_data(test[0:10])
>>>>>>> 155828ac7203f4587a19cd57d15be0b78e641679


    # pool = Pool(processes=8)
    #
    train_name = f'train{tfrecord_version}'
    valid_name = f'valid{tfrecord_version}'
    # test_name = f'test{tfrecord_version}'
    shuffle_to_record(train, labeling_dict,groupfolder, f'../../data/computer_vision_data/{train_name}.tfrecord')
    shuffle_to_record(valid, labeling_dict,groupfolder, f'../../data/computer_vision_data/{valid_name}.tfrecord')
    # shuffle_to_record(test, labeling_dict,groupfolder, f'../../data/computer_vision_data/{test_name}.tfrecord')

