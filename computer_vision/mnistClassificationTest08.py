from tensorflow.python.keras.datasets import mnist

from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Activation # 레이어 추가
from keras import activations,optimizers,metrics
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout

# mnist 테이터셋
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

# reshape 2dim -> 1dim
# convolution
# 4dim으로 바꾸기 개수, 높이, 너비, 채널
x_train = x_train.reshape((60000, 28,28,1)) #(60000, 784)
x_test = x_test.reshape((10000, 28,28,1))

# normalization rgb 1~255 range -> 0 ~ 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# setting graph
nb_classes = 10
model = Sequential()


# convolution layer
# padding = 'valid' (행열수 줄어듬), 'same' ( 행열수 보존 )
# filter 개수는 보통 32,64...
# Conv2D : input_shape -> ( 높이,너비,채널수 )
# 채널수는 흑백 : 1  컬럼 : 3(RGB)
# strides = (1,1) default
# convolution도 여러개로
model.add(Conv2D(filters=32, kernel_size=(3,3), activation=activations.relu, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation=activations.relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation=activations.relu))

# 4차원 데이터를 2차원으로 축소하기
model.add(Flatten())

# full connected
model.add(Dense(units=512, activation=activations.relu))
model.add(Dropout(0.3))
model.add(Dense(units=512,activation=activations.relu))
model.add(Dropout(0.3))
model.add(Dense(units=nb_classes, activation=activations.softmax))

model.summary()

model.compile(loss=losses.categorical_crossentropy,optimizer='rmsprop',metrics=['accuracy'])

print('fitting 중입니다.') # checking workflow output
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

print(model.metrics_names)

print('evaluate 중입니다.') # checking workflow output
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
# model.metric_names와 model.evaluate의 return 결과물은 연관성이 있으니
# 둘의 관계를 공부하자
print('test_acc : %.4f'% test_acc)
print('test_loss : %.4f'% test_loss)
print('-'*50)