# Digit-recognition-using-MNIST-data
The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized.

Tools used : Anaconda 2.1/3.5, Keras
Concepts have to learn : Library keras & Matplot, categorical_crossentropy, optimizer, Dense & DropOut, ReLU and Softmax activation functions

Digit recognition is recognizing the digits from different sources like emails, Bank cheque, papers, Images etc.
Application : It is used to recognize number plates of vehicles, processing bank cheque amounts, Numeric entries in forms filled up by hand and so on.


MNIST DATA is imported directly into code by using keras so,there is no need to create dataset seperately.

To train the model first run "train_model.py"
Once training is completed "mnist_model.h5" file has been created automatically
Now run the "test_mnist_model.py" file by giving path of your input image.

Finally it recognize digits successfully
