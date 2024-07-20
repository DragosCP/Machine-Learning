# example: Let's assume there is a factory line producing wrenches and a second line producing the same wrenches, but we know which ones come from wich line as they are tagged.
# all the wrenches are now pilled together and the workers at the factory go through them to find out the defective wrenches hiding among the pile.
# the question: what is the probability of each factory Line to producing a defective wrench?

# The Bayes Theorem:

# P(A\B) = [ P(B\A) * P(A)] / P(B)

# P = probability
# 

# Let's assume Line 1 : 30 wrenches / hr ; Line 2: 20 wrenches / hr
# Let's assume all produced parts: 1% are defective, we can see 50% came from Line 1 and 50% Line 2
# Q: what is the probability that a part produced by Line 2 is deffective?

# -> P(line1) = 30/50 = 0.6
# -> P(line2) = 20/50 = 0.4
# -> P(Defect) = 1% 
# -> P(Line1|Defect) = 50%  < "|" given some conditions >
# -> P(Line2|Defect) = 50%

# -> P(Defect|Line2) = [ P(Line2|Defect) x P(Defect) ] / P(Line2) = (0.5 x 0.01) / 0.4 = 1.25 %

# Bayes theorem requires independence assumptions which are sometimes incorrect, so it's seen as "naive"

# When we have more than 2 classes it follows a similar/straight process as it always adds up to 1. If we have 3 classes and classify 1 and it's greater than 50% we can assign that class, otherwise we have to calculate for each to assign the classification


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

database = pd.read_csv(r"02_Classification\Social_Network_Ads.csv")
X = database.iloc[:, :-1].values
y = database.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting a new result
predict = classifier.predict(sc.transform([[30,87000]]))
# print(predict)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_concat_y_test = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
# print(y_pred_concat_y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc_sc = accuracy_score(y_test, y_pred)
print(acc_sc)

#results
# [[65  3]
#  [ 7 25]]
# 0.9

# Visualising the Training set results
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
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
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
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()