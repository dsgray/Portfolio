# import necessary modules and packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K

# modules and packages imported in earlier version
#from keras.layers import BatchNormalization
#from keras import regularizers
#from keras.regularizers import l2
#import numpy as np
#import h5py
#import cv2
#import os

# set dimensions of our images.
img_width, img_height = 128, 128

# set directory of images
train_data_dir = '/Users/dgray/Documents/Rutgers/Scripts/20181106 MSC Ki67/cropped and sorted threshold 2/BF for NN/train'
validation_data_dir = '/Users/dgray/Documents/Rutgers/Scripts/20181106 MSC Ki67/cropped and sorted threshold 2/BF for NN/validate'

# set number of training and validation samples
nb_train_samples = 8703 * 2
nb_validation_samples = 8702 * 2

# set number of epochs and batch size
epochs = 25
batch_size = 32

# set shape of input
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# set activation, convolution, and pooling parameters for each layer
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Set parameters of final layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Set hyperparameters
learning_rate = 0.1
decay_rate = 10 * learning_rate / (epochs)
sgd = optimizers.RMSprop(lr=0.001, rho=0.9, decay=decay_rate, clipnorm=5.0)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# This is the augmentation configuration we use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    #rotation_range=180,
    #vertical_flip=True,
    horizontal_flip=True)

# This is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
