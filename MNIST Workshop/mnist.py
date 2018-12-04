'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plot the image and test 
# plt.imshow(x_train[0])
# plt.show()



''' The image_data_format parameter affects how each of the backends treat the data dimensions when working with 
multi-dimensional convolution layers (such as Conv2D, Conv3D, Conv2DTranspose, Copping2D, … and any other 2D or 
3D layer). Specifically, it defines where the 'channels' dimension is in the input data. 
Both TensorFlow and Theano expects a four dimensional tensor as input. But where TensorFlow expects the 
'channels' dimension as the last dimension (index 3, where the first is index 0) of the tensor – i.e. tensor 
with shape (samples, rows, cols, channels) – Theano will expect 'channels' at the second dimension 
(index 1) – i.e. tensor with shape (samples, channels, rows, cols). The outputs of the convolutional layers 
will also follow this pattern.

So, the image_data_format parameter, once set in keras.json, will tell Keras which dimension ordering to use 
in its convolutional layers.
Mixing up the channels order would result in your models being trained in unexpected ways. '''

# print(x_train.shape, 'SHAPE')
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1) # (28,28,1)

x_train = x_train.astype('float32') # Copy of the array, cast to a specified type.
x_test = x_test.astype('float32')

x_train /= 255 # normalize
x_test /= 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical (y_test, num_classes)

model = Sequential()

# in a convelutional network not every layer is connected to the last layer.
# this would be computantially expensive

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', # default go to activation function
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

'''dropouts turn off neurons randomly
 matrice of weights is turned to 0
network is forced to learn new representations for the data it can't always float trough this neuron
prevent overfitting'''

model.add(Dropout(0.25))
model.add(Flatten())

'''
After the convolution and pooling layers, our classification part consists of a few fully connected layers. However, these fully connected layers can only
 accept 1 Dimensional data. To convert our 3D data to 1D, we use the function flatten in Python. This essentially arranges 
our 3D volume into a 1D vector.
'''

model.add(Dense(128, activation='relu'))

'''
The last layers of a Convolutional NN are fully connected layers. Neurons in a fully connected layer have full connections to all the activations in the previous layer.
This part is in principle the same as a regular Neural Network.
'''

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) # softmax for probability distrubition

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# how to save a model
model = model.save('num_reader.model')

# load the model
new_model = keras.models.load_model('num_reader.model')

# make a prediction 
predictions = new_model.predict([x_test]) # predictions always need a list!

print(predictions) # these are probability distrubtions

import numpy as np 
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()