##Adversarial Model--
import os,random
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
from IPython import display
from keras.utils import np_utils
from tqdm import tqdm

K.set_image_dim_ordering('th')
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')
from keras.layers import Activation
import pandas as pd
import glob
from PIL import Image
import os
import time
import datetime
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import keras.callbacks as cb
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

start_time=time.time()
print('Model started at : ' ,  datetime.datetime.now())

#K.set_image_dim_ordering('th')
seed = 1234
np.random.seed(seed)

image_height=350
image_width=350

#read in images/emotions
image_dir_in = 'D:/Erik/Documents/github/mlprojects/data/random/'
legend_file_in= 'D:/erik/documents/github/mlprojects/data/random/legend.csv'

legend = pd.read_csv(legend_in)
image_names = legend['image']
n_images=len(image_names)

emotions=['neutral', 'anger', 'surprise', 'sadness', 'happiness', 'contempt','disgust','fear']

i_emotions=[0,1,2,3,4,5,6,7]
d_emotions1=dict(zip(emotions,i_emotions))
d_emotions2=dict(zip(i_emotions,emotions))
num_classes=8

X_image=[]
y_image=[]

actual_number=0

for i in range(n_images):
    try:

        im=Image.open(image_names[i])
        im = im.convert('L')  # makes it greyscale, else by default it goes to RGB mode with e tuples in color
        sz=im.size
        if sz != (image_height,image_width):
            print('The Image :- ', image_names[i] , ' is of - ', sz , ' dimension, which is incorrect' )
            continue


        im_arr= list(im.getdata())
        im_matrix=np.array(im_arr).reshape(image_height,image_width)   ## im has 3 layers

        im_nm = os.path.basename(image_names[i])
        im_emotion=file[file['image'] == im_nm]['emotion'].values



        if len(im_emotion) == 0:
            continue

    except:
        continue

    X_image.append(im_matrix)
    y_image.append( d_emotions1[im_emotion[0]] )
    actual_number += 1

X_image=np.reshape(X_image,(actual_number, 1, image_height,image_width))
X_image=X_image.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X_image,y_image,test_size=0.2,random_state=99)
y=y_test   ###  saved for comparison purpose later
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 8)
Y_test = np_utils.to_categorical(y_test, 8)

shp = X_train.shape[1:]
print shp

dropout_rate = 0.25
# Optim

opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)

nch = 20



#scale nch down from 200 to 20
nch=20
#scale down the dense layer drastically here, then resize properly with UpSampling2D = (70,70)
# (this is for performance reasons)

g_input = Input(shape=[100])
H = Dense(5*5, init='glorot_normal')(g_input)
H = BatchNormalization(mode=1)(H)
H = Activation('relu')(H)
H = Reshape( [1, 5, 5] )(H)
H = UpSampling2D(size=(70, 70))(H)
H = Convolution2D(nch/2, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=1)(H)
H = Activation('relu')(H)
H = Convolution2D(nch/4, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=1)(H)
H = Activation('relu')(H)
H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

#have to scale down the Convolution layers as well as the Dense layer
d_input = Input(shape=shp)
H = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(8)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val
make_trainable(discriminator, False)
# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


ntrain = 600
trainidx = random.sample(range(0,X_train.shape[0]), ntrain)

XT = X_train[trainidx,:,:,:]

# Pre-train the discriminator network ...
noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
discriminator.fit(X,y, nb_epoch=1, batch_size=32)

y_hat = discriminator.predict(X)


y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)
#Accuracy: 76.50%


# set up loss storage vector
losses = {"d":[], "g":[]}


def train_for_n(nb_epoch=5, plt_frq=50,BATCH_SIZE=16):

    for e in tqdm(range(nb_epoch)):

        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1

        make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1

        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)

        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_loss(losses)
            plot_gen()


train_for_n(nb_epoch=1, plt_frq=25,BATCH_SIZE=16)

K.set_value(opt.lr, 1e-4)
K.set_value(dopt.lr, 1e-5)

train_for_n(nb_epoch=1, plt_frq=10,BATCH_SIZE=16)


def plot_loss(losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()


plot_loss(losses)
