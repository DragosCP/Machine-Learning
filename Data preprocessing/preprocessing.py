
# data pre-processing: import the data, clean it and split it into training & test sets.
# we're running pre-processed data

# Predict which ones of these customers are more likely to purchase a car based on a campaign that the sales division will be running.

# 1. Data Preprocesing tools
#   1.1 Importing the libraries
#   1.2 Importing the dataset
#   1.3 Taking care of missing data
#   1.4 Encoding categorical data
#       1.4.1 Encoding the Independent Variable
#       1.4.2 Encoding the Dependent Variable
#   1.5 Splitting the dataset into the Training set and Test set
#   1.6 Feature Scaling.


# 1.1 importing the libraries

import numpy as np # library that has arrays as inputs
# import matplotlib.pyplot as plt # helps us create charts and graphs 
import pandas as pd # not only import the dataset and pre-process it but also create the matrix of features and the dependent variable vector
from sklearn.impute import SimpleImputer
#sklearn.impute Transformers for missing value imputation.
#SimpleImputer Univariate imputer for completing missing values with simple strategies.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler





# 1.2 Importing the dataset
# read_csv will create a DataFrame exactly like on a database, with columns and rows, using pandas.
dataset = pd.read_csv(r"C:\Users\Dragoss\MachineLearning\Data preprocessing\Data.csv")

# now we need to create the matrix of features and then the dependent variable vector
# In any dataset with which we're going to train a ML model, we have the same entities which are features and the dependent variable vector, which, in our case, is the last column as we are interested to see if customers are going to purchase somethign in the future based on previous information
X = dataset.iloc[:, :-1].values # : taking all the rows from the first column # :-1 except the last column
y = dataset.iloc[:, 1].values

# print(X)
# print(y)




# 1.3 Taking care of missing data from our existing db

# one way is to delete those rows but in our case we will take the average of all other inputs.

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #'mean'=average
imputer.fit(X[:, 1:3]) #in real life we won't probably know if we have gaps or not but we can apply to all numerical columns.
imputer.transform(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)
# print(y)

# output
# [['France' 44.0 72000.0]
#  ['Spain' 27.0 48000.0]
#  ['Germany' 30.0 54000.0]
#  ['Spain' 38.0 61000.0]
#  ['Germany' 40.0 63777.77777777778]
#  ['France' 35.0 58000.0]
#  ['Spain' 38.77777777777778 52000.0]
#  ['France' 48.0 79000.0]
#  ['Germany' 50.0 83000.0]
#  ['France' 37.0 67000.0]]





#   1.4 Encoding categorical data
#       1.4.1 Encoding the Independent Variable (our first column)
#       1.4.2 Encoding the Dependent Variable (the last column)

# One idea is to encode France into 1, Spain into 2 and Germany into 3. The problem is that our ML might interpret that there is a numerical order between these, and the order matters. 
#       1.4.1 The ONE HOT ENCODING is the answer: we turn the country column into 3 separate columns. France would have the vector 1 0 0, Spain 0 1 0 and Germany 0 0 1
#       1.4.2 replace them with 0 and 1s

# we create an object of the column transformer class 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

# output:
# [[1.0 0.0 0.0 44.0 72000.0]
#  [0.0 0.0 1.0 27.0 48000.0]
#  [0.0 1.0 0.0 30.0 54000.0]
#  [0.0 0.0 1.0 38.0 61000.0]
#  [0.0 1.0 0.0 40.0 63777.77777777778]
#  [1.0 0.0 0.0 35.0 58000.0]
#  [0.0 0.0 1.0 38.77777777777778 52000.0]
#  [1.0 0.0 0.0 48.0 79000.0]
#  [0.0 1.0 0.0 50.0 83000.0]
#  [1.0 0.0 0.0 37.0 67000.0]]

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)
# output
# [0 1 0 0 1 1 0 1 0 1]





#   1.5 Splitting the dataset into the Training set and Test set
#   80% of our data will be our training set and 20% will be test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

#   1.6 Feature Scaling.
        # Scaling all your features to make sure they all take values in the same scale.
        
    # Feature scalling: Normalization and Standardization 
# feature scalling is always applied to columns.
# Normalization X' = (X-Xmin) / (Xmax - Xmin) : taking the minimum inside a column, subtracting that minimum  to every single value inside the column and then dividing by the difference between the max and min
# the adjusted values = [0;1]
# Normalization is recommended when you have a normal distribution on most of your features. 

# Standardization X' = (X - u) / sigma : instead of the minimum, we substract the average (u) and divide by the standard diviation (sigma)
# almost all of the adjusted values = [-3;+3]
# Standardization is used all the time

# we always apply feature scaling after splitting the dataset into the training and test sets.
# the test set is going to be a brand new set which we're using to evaluate our ML model.
# if we apply feature scaling before splitting the dataset into training and test sets we will leakage information on the test set

sc = StandardScaler()
# we keep the standardization to the dummy variables alone (the country codes)
# fit method will just get the mean(average) and send the deviation of each your features
# Transform will apply the formula for all the values to be in the same scale.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

#in our case, we apply de standard deviation to the train data (80%) and then we only apply the transform method to the new train data (20%) based on the same scaller used on the training sets. 

print(X_train)
print(X_test)
