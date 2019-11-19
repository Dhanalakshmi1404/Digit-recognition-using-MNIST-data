#from keras.models import Sequential
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from matplotlib import pyplot as plt

img = image.load_img(path="/home/sweety/Desktop/ML/prjcts/digit recognition using mnist data/numbers_recognizer/112.png",grayscale=True,target_size=(28,28,1))
#img = np.invert(img)
img = image.img_to_array(img)
test_img = img.reshape((1,784))

loaded_model= load_model('mnist_model.h5')
class_pred=loaded_model.predict(test_img)
print(class_pred)
print(np.argmax(class_pred[0],axis=0))
img=test_img.reshape(28,28)
plt.imshow(img)
plt.show()
