'''
Based on code found at the following GitHub repos
https://github.com/zatonovo/deep_learning_ex/blob/master/digit_recognition/ex_mnist.py
https://github.com/Vict0rSch/deep_learning/blob/master/keras/feedforward/feedforward_keras_mnist.py
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py#L18
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

To run:
import cnn_mnist
data = cnn_mnist.load_data()
model = cnn_mnist.init_model()
(model, loss) = cnn_mnist.run_network(data, model)
cnn_mnist.plot_losses('loss.png', loss)
Some of the image size parameter will need to be changed to use this with
other types of data.
'''
#from __future__ import print_function
import time
import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
import keras.callbacks as cb
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
KERAS_BACKEND = 'theano'

# model parameters
batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# shape of the image we are loading would need to be changed for other data
input_shape = (1, img_rows, img_cols)

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


# the data, shuffled and split between train and test sets
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print 'Data Loaded'
    return [X_train, X_test, Y_train, Y_test]


def init_model():
    start_time = time.time()
    print 'Compiling Model ... '

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model


def run_network(data=None, model=None, epochs=nb_epoch, batch=batch_size):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Training model...'
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  callbacks = [history],
                  verbose=1, validation_data=(X_test, y_test))

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16)

        print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, history.losses
    except KeyboardInterrupt:
        print 'KeyboardInterrupt'
        return model, history.losses


def predict(model, images):
  """
  Takes an array of images. Obviously dimensions must match training set.
  """
  return model.predict_classes(images)


def display_classes(png, images, classes, ncol=4):
  """
  Draw a number of images and their predictions
  Example:
  images = data[1][:12]
  classes = model.predict_classes('classes.png', images)
  """
  fig = plt.figure()
  nrow = len(images) / ncol
  if len(images) % ncol > 0: nrow = nrow + 1

  def draw(i):
    plt.subplot(nrow,ncol,i)
    plt.imshow(images[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('Predicted: %s' % classes[i])
  [ draw(i) for i in range(0,len(images)) ]
  plt.tight_layout()
  plt.savefig(png)


def plot_losses(png, losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    plt.savefig(png)
