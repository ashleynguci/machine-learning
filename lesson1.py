import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# load the data, X is the whole matrix, while y is the response vector
X, y = load_diabetes(return_X_y=True) 
print(X.shape, y.shape)

# division into training and testing
#permutation Randomly permute a sequence, or return a permuted range.
np.random.seed(42)

#order is a list of dummy value with the length of y
order = np.random.permutation(len(y)) 
#tst is a list of dummy data after sort ( from index 0 to 20)
tst = np.sort(order[:20])
tr = np.sort(order[20:])
print(tst)
print(tr)

print(y)
Xtr = X[tr, :] 
Xtst = X[tst, :] 
Ytr = y[tr]
Ytst = y[tst]

linear_regre = linear_model.LinearRegression(fit_intercept=False)
linear_regre.fit(Xtr, Ytr)

diabetes_y_pred = linear_regre.predict(Xtst)
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(Ytst, diabetes_y_pred)))
