# Logistic regression: predict a CATEGORICAL dependent variable from a number of independent variables. Difference between a linear regression and logistic regression is the fact that we're not predicting a continuous variable but a categorical variable.

# for eg: we want to predict if customers are purchasing a health insurance: Yes/No. We might want to predict this dependent (categorical) variable based on an indipendent variable such an 'age'
# so between yes/no, the model gives us the probability, eg: 
# 35yo -> 42% getting the health insurance
# 45yo -> 81%   
# In real scenarios, everything above 50% will be projected into Yes (a binary 1), under 50% a No (binary 0)

# Of course, we can have multiple independent variables like age, income, education, family/single

# formula:
#
# ln(p/(1-p)) = b0 + b1X1

# p = probability.
# logistic regression curve is also called 'sigmoid curve'


# Maximul Likelihood - finding the best curve to fit our data.
# = for the person of that age, what would the prediction curve look like? What prediction would it made?
# = a person of that age (let's say 20 yo) only has 3% chance of buying health insurance.

# Likelihood = multiplying all the chances we took, for eg:
# = 0.03 x 0.54 x 0.92 x 0.95 x 0.98 x (1-0.01) x (1-0.04) x (1-0.1) x (1x0.58) x (1-0.96) = 0.00019939

# problem: 
# Find out if customer is likely to purchase a car based on age, salary and previous purchase.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





dataset = pd.read_csv(r"Classification/Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split dataset into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# feature scaling: transformic numerical features to a common scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting a new result
predict_example = classifier.predict(sc.transform([[30, 870000]]))
print(predict_example)

# Predicting the Test set results (y_pred) and compare it with real results (y_test)
y_pred = classifier.predict(X_test)
y_pred_and_y_test = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)
print(y_pred_and_y_test)

# Making the Confusion Matrix: the % of results between prediction and test
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

# results
# [[63  5]
#  [ 8 24]]
# 0.87

#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# it takes around 3-5 minutes to plot the first results and 1-2 minutes the second result.