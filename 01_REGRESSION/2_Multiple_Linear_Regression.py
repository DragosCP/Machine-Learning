## formula: y-hat = b0 + b1X1 + b2X2 + ... + bnXn
# y-hat = dependent variable
# b0 = y-intercept (constant)
# b1 = slope coefficient 1
# X1 = Independent variable1 

## Assumptions of Linear Regression
# 1. Linearity - linear relationship between y (our dependent variable) and each X
# 2. Homoscedasticity (cone line spreaded results) - equal variance.
# 3. Multivariate Normality - normality of error distribution
# 4. Independence - of observations. Includes "no autocorrelation" - we don't want patterns.
# 5. Lack of Multicollinearity - predictors are not correlated with each other X1~X2 - which is bad
# 6. The Outlier Check - this is not an assumption, but an "extra check"

## Dummy Variables: states, countries: we allways always omit one dummy variable in our equation on top of the constant b0. We cannot use 2 dummy variables and the b0. Otherwise we would duplicate a variable. This is because D2 = 1 - D1. If we have 100 countries, we only add 99 dummy variables.
# The phenomenon where 1 or several independent variables in a linear regression predict another is called multicollinearity. As a result of this effect, the model cannot distinguish between the effects of D1 from the effect of D2. And therefore it won't work properly.

## P-Values : Statistical Significance.
# It's the point where you get uneasy (intuitivelly) about a prediction. Let's say you flip a coin.
# once for the result Head = 0.5 (50%)
# twice for the result Head = 0.25 (25%)
# third time for Head = 0.12 (12%)
# fourth time for Head = 0.06 (6%)
# __________________________________This is where most of the time the uneasy prediction kicks in, alpha = 0.05 (super suspicios about it). At this point we can say that the coin has Head on both sides.
# fifth time for Head = 0.03 (3%)
# sixth time for Head = 0.01 (1%)


## 5 Methods of building models:
# 1. All-in cases: throw in all your variables, if you've done the model before or you have to
# 2. Backward Elimination
# 3. Forward Selection
# 4. Bidirectional Elimination
# 5. Score Comparison

# Sometimes we hear about Stepwise Regression, which means the steps 2,3 and 4 together.

# 2. Backward Elimination: steps:
#   Step 1: Select a significance level to stay in the model (SL = 0.05)
#   Step 2: Fit the full model with all possible predictors.
#   Step 3: Consider the predictor with the highest P-value
#       If P > SL, go to the next step (4), otherwise go to Finish (our model is ready)
#   Step 4: Remove the predictor
#   Step 5: Fit model without this variable*
#   \-> Step 3

# 3. Forward Selection: steps:
# Step 1: Select a significance level to enter the model (eg SL = 0.05)
# Step 2: Fit all simple regression models y ~ Xn . Select the one with the lowest P-value
# Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
# Step 4: Consider the predictor with the lowest P-value. 
#     If P < SL, go to Step 3, otherwise go to Finish and keep the previou model that was good

# 4. Bidrectional Elimination: combines both step 2 and 3
# Step 1: Select a significance level to enter and to stay in the model eg SLenter = 0.05, SLstay = 0.05
# Step 2: Perform the next step of Forward Selection (new variables must have: P < SLenter to enter)
# Step 3: Perform All steps of Backward Elimination (old variables must have P < SLstay to stay)
# \-> Step 2 and 3 until:
# Step 4: No new variables can enter and no old variables can exit -> Finish

# 5. All Possible Models: (eg for 10 columns in our data it means we have (2 to the power of 10)-1 = 1023 models)
# Step 1: Select a criterion of goodness of fit (eg. Akaike criterion)
# Step 2: Construct all Possible Regression Models: (2 at the power of N)-1 total combinations
# Step 3: Select the one with the best criterion -> Finish.

# https://setosa.io/ev/principal-component-analysis/

# We're going to focus on no 2: Backwards elimination as it is the fastest

# we need to figure out in which startup to invest, out of the 50 companies.
# as we have 3 states, we can apply one hot encoder.

#importing the libraries
import numpy as np
import pandas as pd

# importing the dataset
dataset = pd.read_csv(r"01_REGRESSION\50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
col_tr = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
#col_tr.fit_transform(X)
X = np.array(col_tr.fit_transform(X))

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#in multiple linear regression there is no need to apply feature scalling

#Splitting the dataset into the Training set and Test Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# bare in mind that we now have 4 features instead of one. Therefore we cannot plot a graph like before so we're going to use 2 vectors, the vector of the 10 real profits of the test sets and then the 10 predictive profits of the hole test set to compare them
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# results:
# [[103015.2  103282.38]
#  [132582.28 144259.4 ]
#  [132447.74 146121.95]
#  [ 71976.1   77798.83]
#  [178537.48 191050.39]
#  [116161.24 105008.31]
#  [ 67851.69  81229.06]
#  [ 98791.73  97483.56]
#  [113969.44 110352.25]
#  [167921.07 166187.94]]