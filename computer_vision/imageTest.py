import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array,array_to_img

img_prefix = 'image/'
img_dog = img_prefix + 'mydog.png'

image32 = load_img(img_dog, target_size=(32,32))
print(type(image32))
plt.imshow(image32)
filename = img_prefix + 'dog32.png'
plt.savefig(filename)
print(filename+'저장완료')

image64 = load_img(img_dog, target_size=(64,64))
print(type(image64))
plt.imshow(image64)
filename = img_prefix + 'dog64.png'
plt.savefig(filename)
print(filename+'저장완료')

image85 = load_img(img_dog, target_size=(85,85))
print(type(image85))
plt.imshow(image85)
filename = img_prefix + 'dog85.png'
plt.savefig(filename)
print(filename+'저장완료')

image256 = load_img(img_dog, target_size=(256,256))
print(type(image256))
plt.imshow(image256)
filename = img_prefix + 'dog256.png'
plt.savefig(filename)
print(filename+'저장완료')

# 원본 이미지를 저해상도 이미지로 만들기
def drop_resolution(x, scale=3.0):
    size = (x.shape[0], x.shape[1]) # 원본 사이즈

    # 해상도 떨어뜨린 사이즈
    small_size = tuple(map(int,(x.shape[0]/scale, x.shape[1]/scale)))
    img = array_to_img(x)
    print(small_size)
    small_img = img.resize(small_size,3)
    plt.imshow(small_img)
    filename = img_prefix + 'drop_resolution.png'
    plt.savefig(filename)
    print(filename + '저장 완료')


arr_dog_256 = img_to_array(image256)
print(arr_dog_256)
print(type(arr_dog_256))
print(arr_dog_256.shape)

drop_resolution(arr_dog_256)
print('finished')

