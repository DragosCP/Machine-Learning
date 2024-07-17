# formula:
# y = b0 + b1X1 + b2(X2)square + .. + bn(Xn)to the power of n
# we use it when the data doesn't fit a Linear Regression (the line is not linear (hyperbolic effect), but rather underneath the line initially and then all data goes above the line)
# a good case scenario is used to describe how diseases spread or pandemics and epidemics spread across territory or across population.
# this is called polynomial linear regression - still linear - because it reffers to the coefficients (not the X values). It is was to have a formula where y = (b0 + b1X1) divided by b2X2 or something then it was a different story.
# Polynomial linear regression is actually a special case of the multiple linear regression.


#in this example we're looking at positions/levels and salary to find out, based on the level of experience, the predict previous salary of the candidate.
# let's assume he's been Region Manager (Level 6), but he's been in this position for 2 years, so we assume position 6.5. We are extrapolating, so the salary, should be between 150k and 200k. He mentioned his previous salary was 160k
# We will then compare our predicted salary with his expectation for the new salary.
# this time we will not split the data into training and test sets as we want to maximise the results.

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

# importing the datasets
dataset = pd.read_csv(r"REGRESSION\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#training the linear regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#training the polynomial regression model on the whole dataset
# poly_reg = PolynomialFeatures(degree = 2)
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualising the linear regression results
# we will see that the prediction line (linear regression model) is not well adapted to our case.
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualising the polynomial regression results (for higher resolution and smoother curve)
# we achieve this by not only taking points from 1 to 10 but rather 1.1 1.2 1.3 and so on.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicting a new result with linear regression (input an array [6.5])
print(lin_reg.predict([[6.5]]))
# result: [330378.78787879] - we can see that with simple linear regression the result is far away.

# predict a new result with polynomial regression.
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
# result: [158862.4526515] - the predicition result using polynomial regression is almost perfect in line with what the candidate told us about his previous salary.


