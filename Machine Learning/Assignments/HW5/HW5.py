import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import sklearn.linear_model as linear_model
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics

df = pd.read_csv('ML_HW_Data_CellDNA.csv', header = None)
df.head()

# Binarizing the data
new_cellDna = []
for i in df[13]:
    if i == 0:
        new_cellDna.append(i)
    else:
        i = 1
        new_cellDna.append(i)

df["dependant_variable"] = new_cellDna

# removing the redundant column

df.drop([7, 8, 13], axis=1, inplace=True)
df = df.rename(columns={9: 7, 10: 8, 11: 9, 12: 10})
print(df)
df.info()

# standardizing

X_need_scaling = df.drop(labels=['dependant_variable'], axis=1)
X = preprocessing.scale(X_need_scaling)
y = df["dependant_variable"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
model = LogisticRegression()
model.fit(X_train, y_train)


prediction_test = model.predict(X_test)
print('\n', "Logistic Accuracy: " + str(accuracy_score(y_test, prediction_test)))
print('\n', 'Intercept:', model.intercept_)
print('Co-efficients:', model.coef_)

err = []
err_2 = []
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    err.append(np.average((y_hat - y_test) ** 2))
    err_2.append(model.score(X_test, y_test))
print('\n', "Average Accuracy: " + str(np.average(err_2)))


# Logistic Regression Cross Validation
lrcv = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear', cv=10, max_iter=1500)
lrcv.fit(X_train, y_train)

kf = KFold(n_splits=10, random_state=1234, shuffle=True)
num_cols = 13
offset = 0.000001
number_of_steps = 100
maxC = 1
step = maxC / number_of_steps

#Creating the range of Lambdas

unique_lambdas = np.arange(0 + offset, maxC + step, step)

save_avg_coef = []
save_avg_intercept = []



for the_lambda in unique_lambdas:
    sum_coef = [0] * num_cols
    sum_intercept = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = linear_model.LogisticRegression(C=the_lambda)
        clf.fit(X_train, y_train)
        for i in range(len(clf.coef_[0])):
            sum_coef[i] += clf.coef_[0][i]
        sum_intercept += clf.intercept_

    avg_coef = [0] * num_cols
    for i in range(len(sum_coef)):
        avg_coef[i] = sum_coef[i] / 10
    save_avg_coef.append(avg_coef)
    print('\n', the_lambda, clf.coef_, clf.intercept_)

plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.plot(unique_lambdas, save_avg_coef)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('weights')
plt.title('Lasso coefficients')
plt.tight_layout()
plt.show()

