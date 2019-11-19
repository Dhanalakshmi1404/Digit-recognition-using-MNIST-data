import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt

(train_x, train_y) , (test_x, test_y) = mnist.load_data()
#train_x = train_x.reshape(train_x.shape[0], 28, 28,1)
#test_x = test_x.reshape(test_x.shape[0], 28, 28,1)

train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)
print(type(test_y))
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)

model=Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=SGD(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=32,epochs=15,verbose=0)

model.save('mnist_model.h5')

accuracy=model.evaluate(x=test_x,y=test_y,batch_size=32)
print(('accuracy :',accuracy[1]))
#plt.imshow(train_x[0].reshape(28,28), cmap='gray')
#plt.show()
