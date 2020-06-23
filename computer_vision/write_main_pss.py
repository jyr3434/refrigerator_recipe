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

    groupfolder = 'crl_image_extraction_64'
    # IG = ImgGenerator()
    # img_list = IG.get_path(path)
    # # print(img_list[0])
    # for fp, imgs in img_list:
    #     IG.generator(fp, imgs,path)

    img_list = get_path(groupfolder)
    train, test = seperate_data(img_list)
    train,valid = seperate_data(train)
    labeling_dict = label_dict(test)

    tfrecord_version = '_extraction'
    train = suffle_data(train)
    valid = suffle_data(valid)
    test = suffle_data(test)


    # pool = Pool(processes=8)
    #
    train_name = f'train{tfrecord_version}'
    test_name = f'test{tfrecord_version}'
    valid_name = f'valid{tfrecord_version}'
    shuffle_to_record(train, labeling_dict,groupfolder, f'../../data/computer_vision_data/{train_name}.tfrecord')
    shuffle_to_record(valid, labeling_dict,groupfolder, f'../../data/computer_vision_data/{valid_name}.tfrecord')
    shuffle_to_record(test, labeling_dict,groupfolder, f'../../data/computer_vision_data/{test_name}.tfrecord')

