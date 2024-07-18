# K-NN algorithm:
# let's assume we have 2 categories already present in our dataset: one red card category and one green on the right.
# now we have a new input and we need to figure out in which category it goes to.
# this is where K-NN algorithm will help.

# STEPS:
# 1. Choose the number K of neighbors (common default value = 5)
# 2. Take the K nearest neighbors of the new data point, according to the Euclidean distance (or Manhattan distance or any other)
# 3. Among these K neighbors, count the number of data points in each category.
# 4. Assign the new data point to the category where you counted the most neighbors.

# Euclidean Distance between P1 and P2 = square-root[ (X2-X1)squared + (y2-y1)squared ]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

database = pd.read_csv(r"Classification/Social_Network_Ads.csv")
X = database.iloc[:, :-1].values
y = database.iloc[:, -1].values

# split dataset into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# feature scaling: transformic numerical features to a common scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predict example
predict_example = classifier.predict(sc.transform([[30, 87000]]))

# Predicting the Test set results (y_pred) and compare it with real results (y_test)
y_pred = classifier.predict(X_test)
y_pred_and_y_test = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)
#print(y_pred_and_y_test)

# Making the Confusion Matrix: the % of results between prediction and test
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)
# results 
# [[72  7]
#  [ 4 37]]
# 0.9083333333333333

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# it takes around 3-5 minutes to plot the first results and 1-2 minutes the second result.