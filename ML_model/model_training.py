import numpy as np
import tensorflow as tf
from tensorflow import keras
# import libraries 
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.callbacks import *
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model, load_model

# define the training directory with the images 
train_data_dir  = "/content/test/test"

# set the image size
img_height = 224
img_width = 224

# define some of the model parameters
batch_size = 32

# create an image data generator and apply data augmentation 
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

# training generator 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True,
    class_mode='categorical',
    subset='training') # set as training data

# validation generator 
validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True,
    class_mode='categorical',
    subset='validation') # set as validation data

# define the model architecture, we perform transfer learning using the Xception model trained in colab
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(48, activation='softmax')(x)

for layer in base_model.layers:
    layer.trainable = False

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
print(model.summary())

# Train the top layer of the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

epochs = 50

hist=model.fit(
    train_generator,
    steps_per_epoch = 14320 // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps= (3566 // batch_size))

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# save model
filepath = './ML_model/saved_model'
save_model(model, filepath)