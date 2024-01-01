# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:56:07 2023

@author: HI
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#adds a 2D convolutional layer to the model with 32 filters, each of size 3x3. 
model.add(MaxPooling2D(pool_size = (2, 2)))
#Max pooling is like taking a summary or downsized version of an image. .
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#This helps to increase the robustness and generalization ability of the model by exposing it to different perspectives, distortions, and variations of the input data..
val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 8,
                                                 class_mode = 'binary')
#this code creates a data generator that will load and augment the training images from the specified directory, 
#resize them to a common size, and provide them in batches along with their corresponding labels. 
#This generator can then be used to train a deep learning model on the augmented data

val_set = val_datagen.flow_from_directory('Dataset/val',
                                            target_size = (64, 64),
                                            batch_size = 8,
                                            class_mode = 'binary')

#This generator can be used during the model training process
#to evaluate the performance of the model on the validation data


model.fit_generator(training_set,
                         steps_per_epoch = 50,
                         epochs = 100,
                         validation_data = val_set,
                         validation_steps = 2)

#During training, the model's weights and parameters are updated based on the 
#optimization algorithm specified when compiling the model. 
#The goal is to minimize the loss function and improve the model's performance 
#on both the training and validation datasets.

#By setting the number of epochs, steps per epoch, and validation steps, 
#you can control the duration and granularity of the training process



model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
# JSON files can also be used for configuration files, data storage, 
#and serialization of complex data structures, such as machine learning models, 
#as shown in your code example.



