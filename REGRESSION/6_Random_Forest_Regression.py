# Random Forest is a version of Ensemble Learning (other version such as Gradient boosting) and ensemble learning is when we take multiple algorithms or the same algorithm multiple times and we put them together to make something much more powerful than the original.

# Step 1: Pick at random K data points from the Training set.

# Step 2: Build the Decision Tree associated to these K data points (like a subset)

# Step 3: Choose the number of Ntree of trees you want to build and repeat Steps 1 & 2.

# Step 4: For a new data point, make each one of your Ntree trees predict the value of "y" for the data point in question, and assign the new data point the average across all of the predicted "y" values. End up with hundreds of predictions, that's why the name Forest :)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv(r"REGRESSION\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = RandomForestRegressor(n_estimators=10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Tree Regression!)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Evaluating the Model Performance
# we will apply this method as we use the same dataset for Polynomial, Support Vector, Decision Tree, Random Forest Regressions