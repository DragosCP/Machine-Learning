import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv(r'02_Classification\Select_best_model\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
print("Logistic regression results test & prediction: ")
print(confusion_matrix(y_test, y_pred_log_reg))
print(f"Logistic regression accuracy score is: {accuracy_score(y_test, y_pred_log_reg)}")
print("-------------------------------------------------------------------")

y_pred_knn = classifier_knn.predict(X_test)
print(f"KNN results test & prediction: ")
print(confusion_matrix(y_test, y_pred_knn))
print(f"KNN accuracy score is: {accuracy_score(y_test, y_pred_knn)}")
print("-------------------------------------------------------------------")

y_pred_svm = classifier_svm.predict(X_test)
print(f"SVM results test & prediction: ")
print(confusion_matrix(y_test, y_pred_svm))
print(f"SVM accuracy score is: {accuracy_score(y_test, y_pred_svm)}")
print("-------------------------------------------------------------------")

y_pred_kernel_svm = classifier_kernel_svm.predict(X_test)
print(f"Kernel SVM results test & prediction: ")
print(confusion_matrix(y_test, y_pred_kernel_svm))
print(f"Kernel SVM accuracy score is: {accuracy_score(y_test, y_pred_kernel_svm)}")
print("-------------------------------------------------------------------")

y_pred_gauss = classifier_gauss.predict(X_test)
print(f"Naive Bayes results test & prediction: ")
print(confusion_matrix(y_test, y_pred_gauss))
print(f"Naive Bayes accuracy score is: {accuracy_score(y_test, y_pred_gauss)}")
print("-------------------------------------------------------------------")

y_pred_tree = classifier_tree.predict(X_test)
print(f"Decision Tree results test & prediction: ")
print(confusion_matrix(y_test, y_pred_tree))
print(f"Decision Tree accuracy score is: {accuracy_score(y_test, y_pred_tree)}")
print("-------------------------------------------------------------------")

y_pred_forest = classifier_forest.predict(X_test)
print(f"Random forest results test & prediction: ")
print(confusion_matrix(y_test, y_pred_forest))
print(f"Random forest accuracy score is: {accuracy_score(y_test, y_pred_forest)}")
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