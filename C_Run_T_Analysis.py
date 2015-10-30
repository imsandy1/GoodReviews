import unicodedata
from collections import Counter
from textblob import TextBlob
import timeit, nltk
import numpy as np
from collections import namedtuple
import subprocess
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import sqlite3
import json
import pickle
import scipy.stats
from nltk.corpus import wordnet as wn
from sklearn import linear_model, datasets
from sklearn.decomposition import PCA
import pandas as pd
from pandas import DataFrame
import pandas.io.sql as psql
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from math import sqrt
regressor1 = RandomForestClassifier(n_estimators=101)
from sklearn import svm
from sklearn import cross_validation
from sklearn import decomposition
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.grid_search import GridSearchCV

#Features(Usefulness, ReviewText, SummaryText, Rating, nR_Words, nR_NSWords1, nR_NSWords2, R_POL, R_SUB, nR_LD, nR_LD2, nR_FS, nR_C, nR_N, nR_V, nR_Adj, nR_Adv, nR_G,nS_Words, S_POL, S_SUB, nS_LD, nS_FS, nS_C, nS_N, nS_V, nS_Adj, nS_Adv)
def main():
    N = 100000
    minR = 20
    T1 = 0.2
    T2 = 0.8
    fT1 = T1*100
    fT2 = T2*100
    '''
    a = np.array([[1.0,3.0,5.0],[1.0,3.0,5.0],[2.0,4.0,6.0]])
    print a
    print a-np.mean(a, axis=0)
    print a-np.mean(a, axis=0)/np.std(a, axis=0)
    return
    '''
    conn = sqlite3.connect('../files/T2_Features_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
    c = conn.cursor()
    data_frame = pd.read_sql(sql="SELECT * FROM Features",con=conn)
    
    x = data_frame.iloc[:,3:]
    y = data_frame.iloc[:,0]
    X = x.values
    Y = y.values
    (nr,nc) = X.shape
    for i in xrange(0,nr):
        for j in xrange(2,4):
            X[i,j] = float(X[i,j])
    X = X.astype(float)
    #XX = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    Y1 = Y>T2
    Y1 = Y1.astype(int)
    Y3 = Y<T1
    Y3 = Y3.astype(int)
    Y2a = Y<=T2
    Y2a = Y2a.astype(int)
    Y2b = Y>=T1
    Y2b = Y2b.astype(int)
    Y2 = ((Y2b - Y1) + (Y2a - Y3))/2
    Y2 = Y2.astype(int)
    YY = (2*Y1)+Y2+(0*Y3)
    YY = YY.astype(int)
    XG = X[YY.view(np.ndarray).ravel()==2, :]
    XM = X[YY.view(np.ndarray).ravel()==1, :]
    XB = X[YY.view(np.ndarray).ravel()==0, :]
#Features(Usefulness, ReviewText, SummaryText, Rating 0, nR_Words 1, nR_NSWords1 2, nR_NSWords2 3, R_POL 4, R_SUB 5, nR_LD 6, nR_LD2 7, nR_FS 8, nR_C 9, nR_N 10, nR_V 11, nR_Adj 12, nR_Adv 13, nR_G,nS_Words, S_POL, S_SUB, nS_LD, nS_FS,nS_C, nS_N, nS_V, nS_Adj, nS_Adv)
    
    minR1 = 0
    conn1 = sqlite3.connect('../files/T_AppReviewsFeatures_minR'+str(minR1)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
    c1 = conn1.cursor()
    data_frame1 = pd.read_sql(sql="SELECT * FROM Features",con=conn1)
    a = data_frame1.iloc[:,4:]
    at = data_frame1.iloc[:,1:4]
    b = data_frame1.iloc[:,0]
    A = a.values
    B = b.values
    AT = at.values
    (nr,nc) = A.shape
    for i in xrange(0,nr):
        for j in xrange(3,5):
            A[i,j] = float(A[i,j])
    A = A.astype(float)
    #AA = (A - np.mean(A, axis=0))/np.std(A, axis=0)
    B1 = B>T2
    B1 = B1.astype(int)
    B3 = B<T1
    B3 = B3.astype(int)
    B2a = B<=T2
    B2a = B2a.astype(int)
    B2b = B>=T1
    B2b = B2b.astype(int)
    B2 = ((B2b - B1) + (B2a - B3))/2
    B2 = B2.astype(int)
    BB = (2*B1)+B2+(0*B3)
    BB = BB.astype(int)
    AG = A[BB.view(np.ndarray).ravel()==2, :]
    AM = A[BB.view(np.ndarray).ravel()==1, :]
    AB = A[BB.view(np.ndarray).ravel()==0, :]

    doOVR = True
    if doOVR==True:
        X_train, X_vt, Y_train, Y_vt = train_test_split(X, YY, test_size=0.50, random_state=42)
        X_test, X_val, Y_test, Y_val = train_test_split(X_vt, Y_vt, test_size=0.50, random_state=424242)
        
        Y_train = label_binarize(Y_train, classes=[0, 1, 2])
        Y_test = label_binarize(Y_test, classes=[0, 1, 2])
        Y_val = label_binarize(Y_val, classes=[0, 1, 2])
        n_classes = Y_train.shape[1]
        
        rf = RandomForestClassifier(n_estimators=111)
        clf = OneVsRestClassifier(rf)
        clf.fit(X_train, Y_train)
        
        YY_train = clf.predict(X_train)
        YY_test = clf.predict(X_test)
        YY_val = clf.predict(X_val)
        y_score = clf.predict_proba(X_test)
        y1_score = clf.predict_proba(X_val)
        
        print metrics.classification_report(Y_val, YY_val)
        print metrics.classification_report(Y_test, YY_test)
        ''' 
        clf = OneVsRestClassifier(RandomForestClassifier())
        param_grid = {'n_estimators':[11, 31, 51, 71, 101, 151], 'max_depth':[None, 5, 10, 15, 20, 25], 'min_samples_split':[2, 5, 10],"criterion": ["gini", "entropy"]}
        n_folds= 5
        scoring = 'f1'
        rf = GridSearchCV(clf, cv=n_folds, param_grid=param_grid, scoring=scoring, n_jobs=-1, verbose=-1)
        rf.fit(X_train, Y_train)
        best_params = rf.best_params_
        
        rf = RandomForestClassifier(n_estimators=best_params['n_estimators'], min_samples_split=best_params['min_samples_split'],
                                    max_depth=best_params['max_depth'], criterion=best_params['criterion'])
        clf = OneVsRestClassifier(rf)
        clf.fit(X_train, Y_train)
        YY_train = clf.predict(X_train)
        YY_test = clf.predict(X_test)
        YY_val = clf.predict(X_val) 
        print metrics.classification_report(Y_val, YY_val)
        print metrics.classification_report(Y_test, YY_test)
        '''
        return
    
if __name__ == '__main__':
  main()
'''
rf = RandomForestClassifier()
        clf = OneVsRestClassifier(rf)
        param_grid = {'n_estimators':[11, 31, 51, 71, 101, 151], 'max_depth':[None, 5, 10, 15, 20, 25], 'min_samples_split':[2, 5, 10],"criterion": ["gini", "entropy"]}
        n_folds= 10
        rf = GridSearchCV(clf, cv=n_folds, param_grid=param_grid, scoring=scoring, n_jobs=-1, verbose=-1)
        rf.fit(X_train, Y_train)
        best_params = rf.best_params_
'''
