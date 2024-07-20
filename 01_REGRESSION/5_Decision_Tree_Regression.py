# there are 2 types: Classification Trees and Regression Trees. We will focus on Regression trees.

#                    X1 < 20                           
#                 /           \                       
#                /             \
#              Yes               No
#             /                   \
#       X2 < 200                X2 < 170     
#       /      \               /       \
#     Yes       No            Yes       No
#     /          \           /           \
#  300.5        66.7      X1 < 40       1023
#                         /     \
#                       Yes      No
#                       /         \
#                    -64.1        0.7


# we don't have to apply feature scaling for neither Decision Tree Regression nor Random Forest Regression: 
# That's because the predictions from these models are resulting from successive splits of the data, through the different nodes of our tree.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv(r"01_REGRESSION\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression!)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()