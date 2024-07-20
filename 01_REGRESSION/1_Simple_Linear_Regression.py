# Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If our independent variable is time, then we are forecasting future values, otherwise our model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

## Machine Learning Regression models:

# SIMPLE LINEAR REGRESSION - one indipendent variable (one feature) and one continuous real value to predict
# MULTIPLE LINEAR REGRESSION - same equation but multiple features
# POLYNOMIAL REGRESSION - non-liniar datasets / correlations
# SUPPORT VECTOR for REGRESSION (SVR) - another type of non-linear datasets with non-linear correlations
# DECISION TREE REGRESSION - alternative to predict an outcome for non-linear datasets.
# RANDOM FOREST REGRESSION - same


## Simple Linear Regression
## y = b0 + b1X1
# y - dependent variable which we try to predict
# X1 - independent variable (predictor)
# b0 - y-intercept (constant)
# b1 - slope coefficient

# We're going to predict the ouput of potatos on a farm based on the amount of ferilizer that we use.
# Potatoes[t] = b0 + b1xFertilizer[Kg]
# example of simple linear regression algorithm it came out with values:
# b0 = 8[T]
# b1 = 3[T/Kg]

# We need to apply ORDINARY LEAST SQUARE method to find out which of the slope lines is the best:
# take our data points and project them vertically onto our linear regression line that we are considering.
# on the y axis we're going to have potatos yield in tonnes with yi and yi-hat. E (residual) = yi - yi-hat
# THE BEST EQUATION is where parameters b0 and b1 are such that the sum of the squares of the residuals is minimized. SUM (yi-yi-hat)square is minimized.

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv(r"01_REGRESSION\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into Training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#Training the Simple Linear Regression model on the Training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results
y_prediction = regressor.predict(X_test)

# Visualising the Train set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience (TEST set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction
print(regressor.predict([[12]]))
# predict method always expects a 2D array as inputs.
# 12→scalar 
# [12]→1D array 
# [[12]]→2D array

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
# therefore, the equation of our simple linear regression model is:
        # Salary = 9345.94 x YearsExperience + 26816.19
# Note, coef_ and intercept_ are attributes, different than methods in python, returning a simple value or array of values.
