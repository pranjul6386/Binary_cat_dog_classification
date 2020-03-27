!unzip '/content/drive/My Drive/cat-and-dog.zip'

#importing the libraries

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout ,Flatten ,Dense
from keras import backend as k
from keras.models import load_model
import h5py
import os

#image dimensions and directories

img_wid , img_hit =150,150
training_data = '/content/training_set/training_set'
test_data = '/content/test_set/test_set'
cats_train=os.listdir("/content/training_set/training_set/cats")
dogs_train=os.listdir("/content/training_set/training_set/dogs")
cats_test=os.listdir("/content/test_set/test_set/cats")
dogs_test=os.listdir("/content/test_set/test_set/dogs")
print(len(cats_train)+len(dogs_train))
print(len(dogs_test)+len(cats_test))

#defining trainig ,testind,eppochs,batch_size

nb_train_samples =8007
nb_test_samples =2025
epochs = 50
batch_size = 16
if k.image_data_format() =='channels_first':
  input_shape = (3,img_wid,img_hit)
else:
  input_shape = (img_wid,img_hit,3)
  
#model architecture

model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='RMSprop', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1. /255 , shear_range=0.2 , zoom_range=0.2 , horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1. /255)

#for saving the model

from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,save_weights_only=False, mode='max')
callbacks_list = [checkpoint]

#trainig time

training_set=train_datagen.flow_from_directory('/content/training_set/training_set',target_size=(150,150),batch_size=16,class_mode='binary')
test_set=test_datagen.flow_from_directory('/content/test_set/test_set',target_size=(150,150),batch_size=32,class_mode='binary')
model.fit_generator(training_set,steps_per_epoch=s,epochs=epochs,validation_data=test_set,validation_steps=v,callbacks=callbacks_list)

model.save(filepath) #save the weights


