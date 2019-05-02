# -*- coding: utf-8 -*-

# Pipeline 클래스와 GridSearchCV 클래스를 활용하여
# winequality-red.csv 파일을 분석한 후 결과를 확인하세요.

#from sklearn.datasets import fetch_rcv1
#X, y = fetch_rcv1(return_X_y=True)

#sparse.save_npz("../../data/rcv1/rcv1_X.npz", X)
#sparse.save_npz("../../data/rcv1/rcv1_y.npz", y)

from scipy import sparse

X = sparse.load_npz("rcv1_X.npz")
y = sparse.load_npz("rcv1_X.npz")



