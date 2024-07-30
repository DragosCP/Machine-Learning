# What are Convolutional Neural Networks?
# Step 1 - Convolution Operation, feature detectors, filters, feature maps
# Step 1(b) - ReLU layer - rectified linear unit (linearity is not good, we want more non-linearity in our image recognition)
# Step 2 - Pooling (max, mean pooling)
# Step 3 - Flattening -> from pooled layers to flattened layers 
# Step 4 - Full Connection -> puting everything together
# Summary
# EXTRA: Softmax & Cross-Entropy

# If Geoffrey Hinton is the godfather of artificial neural networks and deep learning, works at Google.
# Yann LeCun is the godfather of convolutional neural networks (a student of Geoffrey Hinton's), works at Facebook.

# Input image -> CNN -> Output label (Image class)
# pixels - 2d array ( 0 to 255): 8 bits of information for 1 pixel = 2 to the power of 8 = 256 -> colour white. 0 is black.

# coloured pixels - 3d array (red, green, blue channels): eg: 255,0,255 (magenta)

## STEPS:

# Step 1: Convolution
# Step 2: Max Pooling
# Step 3: Flattening
# Step 4: Full connection

# We have 4,000 images of dogs and 4,000 images of cats. We also have a test set consisting of 1,000 images of cats and 1,000 of dogs.

## Importing the libraries
import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


#   Part 1 - Data Preprocessing

#       1.1 Preprocessing the Training set # to avoid over fitting otherwise we will get a big difference between training and test sets.
#       image augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,     #this will apply feature scaling to each pixel
    shear_range=0.2,    # the remaining is for image augmentation for the trainset to avoid ofer fitting
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    r'07_Deep_Learning\02_CNN\training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

#       1.2 Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
    r'07_Deep_Learning\02_CNN\test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

#   Part 2 - Building the CNN

#       2.1 Initialising the CNN
cnn = tf.keras.models.Sequential()
#       Step 1: Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#       Step 2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#       Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#       Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

#       Step 4: Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#       Step 5: Output Layer # we do binary activation
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#   Part 3 - Training the CNN
#       3.1 Compiling the CNN (connecting it to an optimiser, loss function and metrics)
cnn.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#       3.2 Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

#   Part 4 - Making a single prediction
test_image = image.load_img(r"07_Deep_Learning\02_CNN\single_prediction\07_02_01.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)

# when we've pre-processed our training and test sets we've actually created batches of images. The CNN was not trained on single images, so we need to specify the extra dimension corresponding to the batch.
test_image = np.expand_dims(test_image, axis=0)
results = cnn.predict(test_image)
# training_set.class_indices will tell us that dog corresponds to 1 and cat to 0
if results[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
