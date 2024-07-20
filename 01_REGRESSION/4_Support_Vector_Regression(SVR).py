# SVR was invented back in the 90 by Vladimir Vapnik and talked about it on his book: The nature of Statistical Learning Theory
# It's called SVR because each point is seen as a vector. Every point outside the tube is a support vector, dictating the formation of the tube. 
# we will cover both Support vector machine and Support vector regression (specifically: Linear) and we will also talk about Kernel SVR

# instead of having a single plot, a single line (linear regression) where we take the ordinary least squares [ SUM(y - y hat)squared -> min ]
# in SVR we have a TUBE:
# this tube is like a margin of error that we are allowing our model to have and not care of any error inside.
# but we do care about the error outside the tube and we will measure the distance (along axis) to the tube (xi). We need them to be minimised. 
# Instead of just the “tube”, the SVR can use the “Epsilon Insensitivity Tube”, that is measured along the y axis.

# advanced study: https://core.ac.uk/download/pdf/81523322.pdf

# Non-linear Support Vector Regression example (slightly more advanced.)
# We're using the same example as per Polynomial regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"01_REGRESSION\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)
y = y.reshape(len(y), 1)
# print(y)


# Feature scaling
from sklearn.preprocessing import StandardScaler
# REMINDER: 
    ##  We have to apply! Feature scaling when the Dependent variable (y) takes super high values with respects to other features, to bring it down in the same range, so we do this to both X and y (+-3)
    ##  We have to apply! Feature scaling whenever we want to split our dataset into the Training set and Test set, but we apply feature scaling after the split, both on X and y.
    ##  We don't apply Feature scaling to some dummy variables resulting from one-hot encoding
    ##  We don't apply Feature scaling when a Dependent variable (y) takes binary values like 0 and 1

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# print(X)
# print(y)

# Training the Support Vector Regression model on the whole dataset
# https://data-flair.training/blogs/svm-kernel-functions/
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# # Predicting a new result : # # prediction on one observation: 6.5
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# #Predicting test results
# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X)).reshape(-1,1))
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y.reshape(len(y),1)),1))

# #Evaluating the Model Performance.
# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)

