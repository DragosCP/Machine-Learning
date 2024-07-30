# Natural Language processing (NLP) is applying ML models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of NLP. Whenever you dictate something into your mobile device that is then converted to text, that's an NLP algorithm in action
# We can also use NLP on a text review to predict if the review is a good one or a bad one. We can use NLP on article to predict some categories of the articles we are trying to segment. We can use it also on a book to predict the genre of the book or to build a machine translator or a speech recognition system or to classify language.
# Most of NLP algorithms are classification models and they include Logistic Regression, Naive Bayes, CART (decision tree), Maximum Entropy (decision trees), Hidden Markov Models which are based on Markov processes.
# A very well-known model in NLP is the "Bag of Words" model. It is used to preprocess the text to classify before fitting the classification algorithms on the observations containing texts.

## We will
# 1. Clean texts to prepare it for ML models
# 2. Create a Bag of Words model
# 3. Apply ML models onto this Bag of Worls model.

# In this example we're not covering DNLP: Seq2Seq

## Classical NLP vs Deep NLP:

## Classical NLP: 
# 1. If/Else Rules (Chatbot) with predefined answers.
# 2. Audio frequency components analysis (Speech Recognition): with predefined data
# 3. Bag-of-words model (Classification): associated words with positive/negative results.

## Deep NLP
# Convolutional Neural Network (CNN) for text Recognition (Classification)
# Seq2Seq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv(r"06_NLP_Natural_Language_Processing\Restaurant_Reviews.tsv", delimiter= '\t', quoting=3)      # this quoting=3 will ignore all double quoting

# Cleaning the texts
import re
import nltk     # removing all the articles "a, the, and" that won't give us any hint if the review is positive or negative
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer      #it takes only the root of a word which will tell us what the word means.
corpus = [] # will contain all cleaned reviews

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # remove all punctuation and replace them by space but ^Not the letters. # in the 3rd argument, we can also use iloc or this new way.
    review = review.lower() # lower case
    review = review.split() # split each review in different words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) # not only join the review but will also add a space
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # the max size of spars matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

classifier_log_reg = LogisticRegression(random_state = 0)
classifier_log_reg.fit(X_train, y_train)

classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)

classifier_kernel_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_kernel_svm.fit(X_train, y_train)

classifier_gauss = GaussianNB()
classifier_gauss.fit(X_train, y_train)

classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_tree.fit(X_train, y_train)

classifier_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_forest.fit(X_train, y_train)

print("-------------------------------------------------------------------")
y_pred_log_reg = classifier_log_reg.predict(X_test)
print("Logistic regression results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_log_reg))
print(f"Logistic regression after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_log_reg)}")
print("-------------------------------------------------------------------")

y_pred_knn = classifier_knn.predict(X_test)
print(f"KNN results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_knn))
print(f"KNN after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_knn)}")
print("-------------------------------------------------------------------")

y_pred_svm = classifier_svm.predict(X_test)
print(f"SVM results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_svm))
print(f"SVM after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_svm)}")
print("-------------------------------------------------------------------")

y_pred_kernel_svm = classifier_kernel_svm.predict(X_test)
print(f"Kernel SVM results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_kernel_svm))
print(f"Kernel SVM after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_kernel_svm)}")
print("-------------------------------------------------------------------")

y_pred_gauss = classifier_gauss.predict(X_test)
print(f"Naive Bayes results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_gauss))
print(f"Naive Bayes after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_gauss)}")
print("-------------------------------------------------------------------")

y_pred_tree = classifier_tree.predict(X_test)
print(f"Decision Tree results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_tree))
print(f"Decision Tree after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_tree)}")
print("-------------------------------------------------------------------")

y_pred_forest = classifier_forest.predict(X_test)
print(f"Random forest results test & prediction after NLP Bag of Words: ")
print(confusion_matrix(y_test, y_pred_forest))
print(f"Random forest after NLP Bag of Words accuracy score is: {accuracy_score(y_test, y_pred_forest)}")
print("-------------------------------------------------------------------")

results = {accuracy_score(y_test, y_pred_log_reg) : "Logistic Regression",
           accuracy_score(y_test, y_pred_knn) : "KNN",
           accuracy_score(y_test, y_pred_svm) : "SVM",
           accuracy_score(y_test, y_pred_kernel_svm) : "Kernel SVM",
           accuracy_score(y_test, y_pred_gauss) : "Naive Bayes",
           accuracy_score(y_test, y_pred_tree) : "Decision Tree",
           accuracy_score(y_test, y_pred_forest) : "Random Forest"}

# bear in mind, didctionaries don't store duplicates, if we print out "results", will only give us 5 elements out of 7, as 2 are duplicates. We should've used a list of dictionaries.
print(f"The best accuracy on this dataset is {max(results)}, comming from {results.get(max(results))} algorithm")
print("-------------------------------------------------------------------")