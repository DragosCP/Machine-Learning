import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataset = pd.read_csv(r'01_REGRESSION\Select_best_model\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
y_train_sc = sc_y.fit_transform(y_train)

regressor_l = LinearRegression()
regressor_l.fit(X_train, y_train)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor_p = LinearRegression()
regressor_p.fit(X_poly, y_train)

regressor_v = SVR(kernel = 'rbf')
regressor_v.fit(X_train_sc, y_train_sc)

regressor_t = DecisionTreeRegressor(random_state = 0)
regressor_t.fit(X_train, y_train)

regressor_f = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_f.fit(X_train, y_train)
print("-------------------------------------------------------------------")
y_pred_l = regressor_l.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_l.reshape(len(y_pred_l),1), y_test.reshape(len(y_test),1)),1))
print("-------------------------------------------------------------------")
y_pred_p = regressor_p.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_p.reshape(len(y_pred_p),1), y_test.reshape(len(y_test),1)),1))

print("-------------------------------------------------------------------")
y_pred_v = sc_y.inverse_transform(regressor_v.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_v.reshape(len(y_pred_v),1), y_test.reshape(len(y_test),1)),1))

print("-------------------------------------------------------------------")
y_pred_t = regressor_t.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_t.reshape(len(y_pred_t),1), y_test.reshape(len(y_test),1)),1))

print("-------------------------------------------------------------------")
y_pred_f = regressor_f.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_f.reshape(len(y_pred_f),1), y_test.reshape(len(y_test),1)),1))

print("-------------------------------------------------------------------")
print(f"Linear Regression: {r2_score(y_test, y_pred_l)}")
print(f"Polynomial Regression: {r2_score(y_test, y_pred_p)}")
print(f"Support Vector Regression {r2_score(y_test, y_pred_v)}")
print(f"Decision Tree Regression {r2_score(y_test, y_pred_t)}")
print(f"Random Forest Regression {r2_score(y_test, y_pred_f)}")

results = {
    r2_score(y_test, y_pred_l): "Linear Regression",
    r2_score(y_test, y_pred_p): "Polynomial Regression",
    r2_score(y_test, y_pred_v): "Support vector Regression",
    r2_score(y_test, y_pred_t): "Decision Tree Regression",
    r2_score(y_test, y_pred_f): "Random forest Regression",
}
print("-------------------------------------------------------------------")
print(f"The best accuracy on this dataset is {max(results)}, comming from {results.get(max(results))} algorithm")
print("-------------------------------------------------------------------")
