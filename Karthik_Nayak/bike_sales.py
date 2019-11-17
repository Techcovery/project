import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np


df = pd.read_csv('bikes.csv')
df = df.drop(columns='date')
print(df.head(25))

X = np.array(df.drop(columns='count'))
Y = np.array(df['count'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

"""####################### Linear Regression ##################"""

clf = LinearRegression()
clf.fit(X_train, Y_train)
linear_reg_accuracy = clf.score(X_test, Y_test)
print("Linear Regression Accuracy is : ", linear_reg_accuracy)

"""####################### Polynomial Regression ##################"""

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_, Y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)
poly_accuracy = clf.score(X_test, Y_test)
print("Polynomial Regression Accuracy : ", poly_accuracy)

"""####################### Ridge Regression ##################"""


from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

X = np.array(df.drop(columns='count'))
Y = np.array(df['count'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = make_pipeline(PolynomialFeatures(degree=1), Ridge())
clf.fit(X_train, Y_train)
ridge_accuracy = clf.score(X_test, Y_test)
print("Ridge Regression Accuracy is : ", ridge_accuracy)

"""####################### Lasso Regression ##################"""


from sklearn.linear_model import Lasso
clf = make_pipeline(PolynomialFeatures(degree=2), Lasso())
clf.fit(X_train, Y_train)
lasso_accuracy = clf.score(X_test, Y_test)
print("Lasso Accuracy is : ", lasso_accuracy)


print("Best Accuracy : %s " %(max(lasso_accuracy, ridge_accuracy,
                                  linear_reg_accuracy, poly_accuracy)))