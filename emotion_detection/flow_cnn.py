import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.optimizers import RMSprop
import keras.callbacks as cb
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def init_model():
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    nb_classes = 8

    start_time = time.time()
    print 'Compiling Model ... '

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(3, 50, 50)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(2*nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(.25))
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #Train = 83.52% Test = 83.74%
    # adagrad: Train = 0.7608 Test = 0.7859
    # adadelta: Train = 0.8152  Test = 0.8270
    # adam: Train = 0.8264 Test = 0.8353
    # adamax: Train = .8 Test = .8178
    # nadam: Train = 0.8231 Test = 0.8320
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=['accuracy'])

    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model


def run_network(model, epochs=16, batch=61):
    start_time = time.time()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    train_datagen2 = ImageDataGenerator(
        rescale=1./255
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    history = LossHistory()

    train_generator = train_datagen2.flow_from_directory(
        'D:/Erik/Documents/GitHub/MLProjects/data/flow/train',
        target_size=(50, 50),
        batch_size=batch,
        seed=1234)

    test_generator = test_datagen.flow_from_directory(
        'D:/Erik/Documents/GitHub/MLProjects/data/flow/test',
        target_size=(50, 50),
        batch_size=batch,
        seed=1234)

    print 'Training model...'

    model.fit_generator(
        train_generator,
        samples_per_epoch=9638,
        nb_epoch=epochs,
        validation_data=test_generator,
        nb_val_samples=2409,
        callbacks=[history])

    #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
     #             callbacks = [history],
     #             verbose=1, validation_data=(X_test, y_test))

    print "Training duration : {0}".format(time.time() - start_time)

    score = model.evaluate_generator(test_generator, val_samples=50, max_q_size=10)

    print "Network's test score [loss, accuracy]: {0}".format(score)

    model.save_weights('D:/Erik/Documents/GitHub/MLProjects/model/flow.h5')

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
