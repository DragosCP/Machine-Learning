## We will learn:
#
# 1. The Neuron
# 2. The Activation Function (sits inside the Neuron): Threshold Activation Function, Sigmoid Activation Function, Rectifier function, Hyperbolic Tangent function
# 3. How do Neural Networks work? (example)
# 4. How do Neural Networks learn?
# 5. Gradient Descent
# 6. Stochastic Gradient Descent
# 7. Backpropagation

# In neural netwrok there is a processed called forward progagation, where information is entered into the input layer and then is propagated forward to get our y hats, our output values and then we compare those to the actual values y that we have in our training set and then we calculate the errors. Then the errors are back propagated through the network in the opposite direction.
# And that allows us to train the network by adjusting the weights.
# Now, Back propagation is an advanced algorithm driven by sophisticated maths, which allows us to adjust all of the weights, simultaneously, at the same time. Because of this, it knows all our weights and which is responsible for what.
# So basically we know which part of the error, each of our weights in the neural network, is responsible for.

## Steps

# 1. Randomly initialise the weights to small numbers close to 0 (but not 0)

# 2. Input the first observation of our dataset in the input layer, each feature in one input node.

# 3. Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations untill getting the predicted result y.

# 4. Compare the predicted result to the actual result. Measure the generated error.

# 5. Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights. 

# 6a. Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning) or
# 6b. Repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch Learning)

# 7. When to whole training set passed through the ANN, that makes an epoch. Redo more epochs
    # so the ANN gets better and better and constantly adjust itself as we minimise the cost function.




# Restricted Boltzmann or deep Boltzmann machines are great examples of computational graph, they're not covered in this course



### PART 1 - Data Preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
# pytorch is another great library to build NNs

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# 1.1 Importing the dataset
dataset = pd.read_csv(r"07_Deep_Learning\01_ANN\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# 1.2 Encoding categorical data:
# Gender (label encoding)
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Geography (one hot encoding): remember dummy variables are moved to first columns of our matrix feature
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 1.3 Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1.4 Feature Scaling: in ANN we need feature scalling for the entire data.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### PART 2 - Building the Artificial Neural Network

# 2.1 Initialising the ANN as a sequance of layers.
ann = tf.keras.models.Sequential()

# 2.2 Adding the input layer and the first hidden layer: we're going to use the famous Dense class
# we add either a hidden layer or a dropout layer, which allows to prevent over fitting or (we will see with convolution neural network) a conv2D layer but for now we need a fully connected layer.
# activation 'relu' as we have a Rectifier activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2.3 Adding the second layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2.4 Adding the output layer # units means the what dimension we are in and because the results (Exited column) has binary outcome, 0 or 1, the dimension is 1, 1 output neuron
# the activation function is a SIGMOID activation for the output layer as it's doing the prediction and probability for customer to leave or stay; for predictions for more than one category (2+) we need to use SOFT MAX (not sigmoid)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


### PART 3 - Training the ANN

# 3.1 Compiling the ANN with an optimizer, a loss function, and a metric which will be the accuracy as we're doing some classification
# Optimizer we choose Atom (very performance) that can perform Stochastic Gradient Descent which will allow to update the weights in order to reduce the loss error between our predictions and the real results
# The loss function - to compute the difference between the prediction and real results and accuracy, in our case, being binary outcome, we choose binary_crossentropy, for non binary we just need crossentropy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# 3.2 Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

### PART 4 - Making the predictions and evaluating the model
# 4.1 Predicting the result of a single observation


# Use our ANN model to predict if the customer with the following informations will leave the bank: 
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000


prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
# prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]) > 0.5)
print(prediction)

# 4.2 Predicting the Test set result
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# 4.3 Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc_sc = accuracy_score(y_test, y_pred)
print(acc_sc)