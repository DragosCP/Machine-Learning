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

# Step 1: Convolution (+ ReLU Layer)
# Step 2: Max Pooling
# Step 3: Flattening
# Step 4: Full connection

# STEP 1: Convolution: 
# Input image comes as a matrice of 0s and 1s and then we have a Feature Detector (or Kernel or Filter), which is a matrice of 3x3 or 5x5 or 7x7 and then we multiply the values from input image with the Feature detector and the results are going into a Feature map: 0x0 = 0, 0x1 = 0, 1x1 = 1, if we have 2 x 1x1 = 2 and so on. The feature detector moves one column at a time and gets the updated results in the feature map (sometimes called activation map).
# The result is: we've reduced the size of the image (easier to process, faster) and we're detecting some parts of the image that are integral (which means it matches exactly with our Feature Detector). 
# in reality it is the same: we've looking at an image, let's say a cat, and if we see the face of the cat, or the shape, nose, basically we're looking at features, we realise it is a cat without looking at the entire pixels of the image.
# Also, the network decides on multiple feature maps to preserve lots of information (this is done through training) and applies different filters for each feature map
# gimp.org -> free tool to adjust images.

# 1b - ReLU Layer
# basically we apply our Rectifier function (from ANN) -> we want to increase non-linearity in our CNN (break up the linearity) and the reason for this is that images themselves are highly non-linears (different borders, colours, elements) https://arxiv.org/pdf/1609.04112 
# https://arxiv.org/pdf/1502.01852

# STEP 2: MAX POOLING
# spatial invariance - allows the NN to locate features accurately (tilted features, too close, further apart, relative to each other, disorted)
# we are working with convolution (Feature) map and we apply Max (or Min) Pooling => we take a box of 2x2 pixels (or larger) and we place it in the top left hand corner and we find the max value in that box and then we record only that value (disregarding the other 3) and then we move 2 pixels on the right and record again, and so on, until our POOLED FEATURE MAP is done.
# disregarding 3 pixels out of 4 (75%) of the information that is not the important feature, and, because we're taking the max value, we're accounting for any distorsion + we're reducing the size again by reducing the number of parameters that are going into our Neural Network
# All this adds up to preventing overfitting !!
# https://www.ais.uni-bonn.de/papers/icann2010_maxpool.pdf

# STEP 3: FLATTENING
# we're taking our pooled feature map 3x3 and turn it into a Column 1x9 - input layer of ANN

# STEP 4: FULL CONNECTION
# now we're ready to add everything to Artificial Neural Network ->
# -> Input layer (X1, X2,.. Xn) -> Fully Connected Layer (simillar to Hidden Layers in ANN where there don't have to be fully connected)-> Output Layer (output value)
# -> The ANN from this point is going to create more feature layers
# -> in the ANN if the result is not satisfied, we call a CROSS function in ANN (we used means square error there) or a LOSS function (CNN -> we use a cross entropy -> and then error is calculated, then it is back propagated through the network and the weigths and feature detectors (matrices we've put on top) are adjusted. All these are done with a lot of science in the background, math, through gradient descent and back propagation.

# https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

## Softmax & Cross-Entropy
# if we have 2 outputs, dog and cat, and the weight of the dog is 0.95 and the weight of the cat is 0.5, the dog and the cat won't add up to 1. 
# 
# The Softmax function, or the normalized exponential function, is a generalization of the logistic function that "squashes a k-dimensional vector of arbitrary real values to a k-dimensional vector of real values in the range of 0-1 that all add up to 1. (wiki)

# The Cross-entropy function:
# (something like the mean squared error function which we've used as the cost-function for assessing (goal to minimize the MSE) of our neural network performance)
# it's called the loss function in CNN - that we want to minimise in order to maximize the performance of our neural network.

# http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/


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
