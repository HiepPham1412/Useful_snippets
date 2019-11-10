import pandas as pd
import numpy as np
import itertools
import multiprocessing
from multiprocessing import Pool
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc


log_reg = LogisticRegression()
classifiers = [log_reg]


df = pd.read_csv('bank.csv')
X = df.loc[:, ['age', 'balance', 'pdays']]
y = df.loc[:, 'y'].map({'yes': 1, 'no': 0})
y = np.ravel(y)



sss = StratifiedShuffleSplit(n_splits = 4, test_size = 0.1, random_state = 0)
cv_index = [(i, j) for i, j in sss.split(X, y)]
params = list(itertools.product(cv_index, classifiers), X, y)


def cv_test(params):
    # global X
    # global y
    train_index = params[0][0]
    test_index = params[0][1]
    clf = params[1]
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_probas = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probas[:,1])
    auc_score = auc(fpr, tpr)
    return [name, acc, loss, auc_score]

p = Pool(processes = 4)
start = time.time()
res = p.map(cv_test, params)
p.close()
p.join()
print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
