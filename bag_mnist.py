
'''
K-Fold Cross Validation MNIST Classifiier based on code found at:
https://github.com/zatonovo/deep_learning_ex/blob/master/digit_recognition/ex_mnist.py
https://github.com/Vict0rSch/deep_learning/blob/master/keras/feedforward/feedforward_keras_mnist.py
https://www.kaggle.com/danijelk/allstate-claims-severity/keras-starter-with-bagging-lb-1120-596
K-Fold with SciKit-Learn
http://scikit-learn.org/0.17/modules/generated/sklearn.cross_validation.KFold.html
'''

## import libraries
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

## Batch generators ############################################################
# For this method we will be using the a k fold corss validation where the data
# is sampled for each batch from a random shuffle of the intput data.

def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :]
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


## Loss History ###############################################################
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


## Load Data ###################################################################
# read data from the mnist data set that is included with the keras package.
def load_data():
    print ('Loading data...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))

    print ('Data loaded.')
    return [X_train, X_test, y_train, y_test]

## Build the Neural Network ####################################################
## neural net
def init_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(500, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms,
      metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model

## Run the Network ##############################################################

def run_network(data=None, model=None, epochs=1, batch=256, nfolds = 5):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        # cv-folds this creates a n-fold cross validation set defaults to 5-fold

        # This splits the training data into n-folds

        y=y_train
        folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

        ## train models
        i = 0
        nbags = 30
        pred_oob  = np.zeros(X_train.shape[0])
        pred_test = np.zeros(X_test.shape[0])


        for (inTr, inTe) in folds:
          xtr = X_train[inTr]
          ytr = y_train[inTr]
          xte = X_train[inTe]
          yte = y_train[inTe]
          pred = np.zeros(xte.shape[0])

          # We will now feed the k-fold training data into the model for the
          # number of times specified by the nbags passed to the run_network()
          for j in range(nbags):
            model = init_model()
            fit = model.fit_generator(generator = batch_generator(xtr, ytr, 200, True),
                                      nb_epoch = epochs,
                                      samples_per_epoch = xtr.shape[0],
                                      verbose = 0)
            pred += model.predict_generator(generator = batch_generatorp(xte, 200, False), val_samples = xte.shape[0])[:,0]

            pred_test += model.predict_generator(generator = batch_generatorp(X_test, 200, False), val_samples = X_test.shape[0])[:,0]

            pred /= nbags

          pred_oob[inTe] = pred

          yscore= [1] * len(yte)
          score = mean_absolute_error(yscore, pred)
          i += 1
          print('Fold ', i, '- MAE:', score)


        y=[1]*len(pred_oob)
        print('Total - MAE:', mean_absolute_error(y, pred_oob))
        pred_test /= (nfolds*nbags)


    except KeyboardInterrupt:
      print (' KeyboardInterrupt')

      return model, history.losses


def predict(model, images):
  """
  Takes an array of images. Obviously dimensions must match training set.
  """
  return model.predict_classes(images)



def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()
