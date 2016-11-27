'''
Based on code found at the following GitHub repos
https://github.com/zatonovo/deep_learning_ex/blob/master/digit_recognition/ex_mnist.py
https://github.com/Vict0rSch/deep_learning/blob/master/keras/feedforward/feedforward_keras_mnist.py
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py#L18
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

@author: Erik Nylander

To run:
import emotion_detection_cnn
data = emotion_detection_cnn.load_data()
model = emotion_detection_cnn.init_model()
(model, loss) = emotion_detection_cnn.run_network(data, model)
emotion_detection_cnn.plot_losses('loss.png', loss)
Some of the image size parameter will need to be changed to use this with
other types of data.
'''
#from __future__ import print_function
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.models import Sequential
import keras.callbacks as cb
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.cross_validation import KFold, train_test_split
from PIL import Image
K.set_image_dim_ordering('th')

# model parameters
batch_size = 64
nb_epoch = 12

emotions = ['neutral', 'anger', 'surprise', 'sadness', 'happiness', 'contempt', 'disgust', 'fear']
i_emotions = [0,1,2,3,4,5,6,7]
d_emotions1 = dict(zip(emotions, i_emotions))
d_emotions2 = dict(zip(i_emotions, emotions))

nb_classes=len(i_emotions)

# input image dimensions
image_height = 64
image_width = 64
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (8, 8)
# shape of the image we are loading would need to be changed for other data
input_shape = (1, image_height, image_width)

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

image_dir_in = 'D:/Erik/Documents/GitHub/MLProjects/data/images/'
legend_in = 'D:/Erik/Documents/GitHub/MLProjects/data/legend.csv'

# the data, shuffled and split between train and test sets
def load_data():
    # Loading image names
    legend = pd.read_csv(legend_in)
    image_names = legend['image']
    n_images = len(image_names)

    X_image = []
    y_image = []

    actual = 0

    for i in range(n_images):
        im = Image.open(image_dir_in + image_names[i])
        # makes it greyscale, else by default it goes to RGB mode with e tuples in color
        im = im.convert('L')
        sz = im.size
        if sz != (image_height, image_width):
            print('The Image :- ', image_names[i], ' is of - ', sz, ' dimension, which is incorrect')
            continue

        actual += 1

        im_arr = list(im.getdata())
        im_matrix = np.array(im_arr).reshape(image_height, image_width)   ## im has 3 layers
        X_image.append(im_matrix)

        #im_nm = os.path.basename(image_names[i])
        im_emotion = legend[legend['image'] == image_names[i]]['emotion'].values
        y_image.append(d_emotions1[im_emotion[0]])

    X_image = np.reshape(X_image, (actual, 1, image_height, image_width))
    X_image = X_image.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X_image, y_image, test_size=0.2, random_state=1234) #, random_state=1234 83.4%

    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 8)
    Y_test = np_utils.to_categorical(y_test, 8)
    print 'Data Loaded'
    return [X_train, X_test, Y_train, Y_test]


def init_model():
    start_time = time.time()
    print 'Compiling Model ... '

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(1, 64, 64)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Convolution2D(nb_filters/2, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))

    model.add(Flatten())
   
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
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
    plt.subplot(nrow, ncol,i)
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
