#source ~/tensorflow/bin/activate

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras.regularizers import l2
from keras import backend as K
import numpy as np
import h5py
import cv2
import os

#print('Directory name: ')
#dir = input()

# dimensions of our images.
img_width, img_height = 128, 128
train_data_dir = '/Users/dgray/Documents/Rutgers/Scripts/20181106 MSC Ki67/cropped and sorted threshold 2/BF for NN/train'
validation_data_dir = '/Users/dgray/Documents/Rutgers/Scripts/20181106 MSC Ki67/cropped and sorted threshold 2/BF for NN/validate'
nb_train_samples = 8703 * 2
nb_validation_samples = 8702 * 2
epochs = 25
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
#model.add(Conv2D(32, (3,3), input_shape=input_shape, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
#model.add(Conv2D(32, (3, 3), input_shape=input_shape, W_regularizer=l2(0.1)))
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))

#model.add(Conv2D(32, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
#model.add(Conv2D(32, (3, 3), W_regularizer=l2(0.1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))

#model.add(Conv2D(64, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
#model.add(Conv2D(64, (3, 3), W_regularizer=l2(0.1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))


#model.add(Conv2D(64, (3, 3), W_regularizer=l2(0.1)))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

#model.add(Conv2D(128, (3, 3), W_regularizer=l2(0.1)))
#model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))



#model = Sequential()
#model.add(Conv2D(16, (3, 3), input_shape=input_shape)) #32
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(16, (3, 3))) #32
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3))) #32
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3))) #64
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3))) #128
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64)) #64
#model.add(BatchNormalization())
#model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#model.compile(loss='binary_crossentropy',
#              optimizer='RMSprop',
#              metrics=['accuracy'])

learning_rate = 0.1
decay_rate = 10 * learning_rate / (epochs)
sgd = optimizers.RMSprop(lr=0.001, rho=0.9, decay=decay_rate, clipnorm=5.0)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    #rotation_range=180,
    #vertical_flip=True,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
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

#tf.keras.preprocessing.image.load_img(
#    path,
#    grayscale=False,
#    target_size=None,
#    interpolation='nearest'
#)

###############

#dir = 'decaytimes10_20190110'
#model.save('classify_' + dir + '.h5')

#test_directory = '/Users/dgray/Documents/Rutgers/Scripts/20181227 Ki67 MSCs in flow/validate/low'

#for root, dirs, files in os.walk(test_directory):
    #for file in files:
        #img = cv2.imread(test_directory + '/' + file)
        #img = np.expand_dims(img, axis=0)
        #if img.shape == (1, 128, 128, 3):
            #img = 1./255 * img
            #img_class = model.predict_proba(img)
            ##prediction = img_class[0]
            ##print(prediction)
            #print(img_class)
        #else:
            #continue
