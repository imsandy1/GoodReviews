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
import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sknn.mlp import Classifier, Layer
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
#regressor1 = RandomForestRegressor(n_estimators=101)
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
def isFloat(f):
    try:
        float(f)
        return True
    except ValueError:
        return False
def isInt(f):
    try:
        int(f)
        return True
    except ValueError:
        return False


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
    #print data_frame.head()
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
    makePlots = False
    if makePlots==True:
        pi = [0, 1, 4, 6, 8, 10, 12, 13]
        pmin = [1.0, 0.0, -1.0, 0.2, 0.0, 0.0, 0.0, 0.0]
        pmax = [5.0, 2000.0, 1.0, 1.0, 100.0, 8.0, 8.0, 8.0]
        pxt = ['Rating', 'Number of words', 'Polarity', 'Lexical diversity', 'Number of sentences', 'Number of nouns', 'Number of adjectives', 'Number of adverbs']
        bw = [0.5, 50, 0.05, 0.05, 2, 1, 1, 1]
        for i in xrange(0,len(pi)):
            fig1 = plt.figure(figsize=(10,10))
            plt.hist(XG[:,pi[i]],bins=np.arange(min(X[:,pi[i]]), max(X[:,pi[i]]) + bw[i], bw[i]),label='Helpful',histtype='stepfilled',color='b')
            #plt.hist(XM[:,pi[i]],bins=np.arange(min(X[:,pi[i]]), max(X[:,pi[i]]) + bw[i], bw[i]),alpha=0.5,label='Middle',fill=False,histtype='step',linewidth=3)
            plt.hist(XB[:,pi[i]],bins=np.arange(min(X[:,pi[i]]), max(X[:,pi[i]]) + bw[i], bw[i]),alpha=0.2,label='Not helpful',histtype='stepfilled',color='r')
            plt.xlabel(pxt[i], fontsize=30)
            plt.ylabel("N (log scale)", fontsize=30)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.yscale('log')
            plt.xlim([pmin[i],pmax[i]])
            legend = plt.legend(loc='upper right', shadow=False)
            for label in legend.get_texts():
                label.set_fontsize('xx-large')
            fig1.canvas.manager.window.attributes('-topmost', 1)
            fig1.canvas.manager.window.attributes('-topmost', 0)
            pylab.savefig("../plots/June19/"+str(pi[i])+".png",bbox_inches='tight')

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
        #X_vt = (X_vt - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
        X_test, X_val, Y_test, Y_val = train_test_split(X_vt, Y_vt, test_size=0.50, random_state=424242)
        X_test = (X_test - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
        X_val = (X_val - np.mean(X_train, axis=0))/np.std(X_train, axis=0) 
        
        AA = (A - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
        X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
        #print Y_train[0]
        Y_train = label_binarize(Y_train, classes=[0, 1, 2])
        #print Y_train[0]
        Y_test = label_binarize(Y_test, classes=[0, 1, 2])
        Y_val = label_binarize(Y_val, classes=[0, 1, 2])
        n_classes = Y_train.shape[1]
        '''
        md = np.array([7,11,15,19,25])
        ne = np.array([11,41,71,101,131,161,201])
        for i in xrange(0,len(md)):
            for j in xrange(0,len(ne)):
                print md[i],'    ', ne[j]
                rf1 = RandomForestClassifier(n_estimators=ne[j],max_depth=md[i])
                clf1 = OneVsRestClassifier(rf1)
                print 'Fitting.....'
                clf1.fit(X_train, Y_train)
                YG_train = clf1.predict(X_train)
                YG_test = clf1.predict(X_test)
                
                print metrics.classification_report(Y_test, YG_test)
        '''
        rf = RandomForestClassifier(n_estimators=111)
        clf = OneVsRestClassifier(rf)
        clf.fit(X_train, Y_train)
        #print rf.oob_score_
        YY_train = clf.predict(X_train)
        YY_test = clf.predict(X_test)
        YY_val = clf.predict(X_val)
        y_score = clf.predict_proba(X_test)
        y1_score = clf.predict_proba(X_val)
        print Y_test[0], '   ', YY_test[0], '  ', y_score[0]
        mc = np.zeros(3)
        yr,yc = Y_test.shape
        for i in xrange(0,yr):
            if Y_test[i,2]==1:
                mc = mc + YY_test[i,:]
        print mc
        mc = np.zeros(3)
        yr,yc = Y_val.shape
        for i in xrange(0,yr):
            if Y_val[i,2]==1:
                mc = mc+ YY_val[i,:]
        print mc
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        th = dict()
        fpr1 = dict()
        tpr1 = dict()
        roc_auc1 = dict()
        th1 = dict()
        tROC = []
        for i in range(n_classes):
            fpr[i], tpr[i], th[i] = roc_curve(Y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        for i in range(n_classes):
            fpr1[i], tpr1[i], th1[i] = roc_curve(Y_val[:, i], y1_score[:, i])
            roc_auc1[i] = auc(fpr1[i], tpr1[i])

        
        fpr["micro"], tpr["micro"], th["micro"] = roc_curve(Y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr1["micro"], tpr1["micro"], _ = roc_curve(Y_val.ravel(), y1_score.ravel())
        roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])
        minI = np.array([])
        for j in range(n_classes):
            dist = np.array([])
            for i in xrange(0, len(fpr[j])):
                dist = np.append(dist, np.sqrt( (fpr[j][i]*fpr[j][i]) + ((tpr[j][i]-1)*(tpr[j][i]-1)) ))
            minI =  np.append(minI, np.where(dist==np.min(dist)))
        print minI, '   ', th[0][minI[0]], '   ', th[1][minI[1]], '   ', th[2][minI[2]]
        thV = np.array([th[0][minI[0]],th[1][minI[1]],th[2][minI[2]]])
        yvr,yvc = y1_score.shape
        YYY_val = YY_val[:]
        for i in xrange(0,yvr):
            YYY_val[i,:] = y1_score[i,:]>thV
        print metrics.classification_report(Y_val, YY_val)
        print metrics.classification_report(Y_val, YYY_val)
        
        fig = plt.figure()
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        '''
        print len(fpr["micro"]),'   ', len(fpr[0]),'   ', len(fpr[1]),'   ', len(fpr[2])
        for i in xrange(0,len(fpr["micro"])):
            if (fpr["micro"][i] + tpr["micro"][i])>0.92 and (fpr["micro"][i] + tpr["micro"][i])<1.08:
                tROC.append(th["micro"][i])
        print 'AAAAA\n', tROC
        '''

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('')
        plt.legend(loc="lower right")
        pylab.savefig('../plots/June19/ROC_test.png')

        fig1 = plt.figure()
        plt.plot(fpr1["micro"], tpr1["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc1["micro"]))
        for i in range(n_classes):
            plt.plot(fpr1[i], tpr1[i], label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc1[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('')
        plt.legend(loc="lower right")
        #plt.show()
        #fig.canvas.manager.window.attributes('-topmost', 1)
        #fig.canvas.manager.window.attributes('-topmost', 0)
        pylab.savefig('../plots/June19/ROC_val.png')

        precision = dict()
        recall = dict()
        average_precision = dict()
        precision1 = dict()
        recall1 = dict()
        th2 = dict()
        average_precision1 = dict()
        for i in range(n_classes):
            precision[i], recall[i], th2[i] = precision_recall_curve(Y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        '''
        tPR = [] 
        for i in xrange(0,len(th2[2])):
            if (precision[2][i] - recall[2][i])>-0.01 and (precision[2][i] - recall[2][i])<0.01:
                tPR.append(th[2][i])
                
        print 'BBBBB\n', tPR
        '''
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

        for i in range(n_classes):
            precision1[i], recall1[i], _ = precision_recall_curve(Y_val[:, i],
                                                                y1_score[:, i])
            average_precision1[i] = average_precision_score(Y_val[:, i], y1_score[:, i])
        
        precision1["micro"], recall1["micro"], _ = precision_recall_curve(Y_val.ravel(), y1_score.ravel())
        average_precision1["micro"] = average_precision_score(Y_val, y1_score, average="micro")
        
        fig2 = plt.figure()
        plt.plot(recall["micro"], precision["micro"],
                 label='micro-average precision recall curve (area = {0:0.2f})'
                 ''.format(average_precision["micro"]))
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], label='Precision recall curve of class {0} (area = {1:0.2f})'
                     ''.format(i, average_precision[i]))
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.title('')
        plt.legend(loc="lower right")
        pylab.savefig('../plots/June19/PR_test.png')
        
        fig3 = plt.figure()
        plt.plot(recall1["micro"], precision1["micro"],
                 label='micro-average precision recall curve (area = {0:0.2f})'
                 ''.format(average_precision1["micro"]))
        for i in range(n_classes):
            plt.plot(recall1[i], precision1[i], label='Precision recall curve of class {0} (area = {1:0.2f})'
                     ''.format(i, average_precision1[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.title('')
        plt.legend(loc="lower right")
        pylab.savefig('../plots/June19/PR_val.png')
        
        print accuracy_score(Y_train, YY_train)
        print accuracy_score(Y_test, YY_test)
        print accuracy_score(Y_val, YY_val)
        print metrics.classification_report(Y_train, YY_train)
        print metrics.classification_report(Y_test, YY_test)
        print metrics.classification_report(Y_val, YY_val)
        return
        BB = label_binarize(BB, classes=[0, 1, 2])
        BBB = clf.predict(AA)
        print metrics.classification_report(BB, BBB)
        conn2 = sqlite3.connect('../files/TTF20_App.db')
        c2 = conn2.cursor()
        #c2.execute('''DROP TABLE WebApp''')
        c2.execute('''CREATE TABLE IF NOT EXISTS WebApp(Prediction, Title, Review, Summary, Rating, nR_Words)''')
        bbbr,bbbc = BBB.shape
        BBB2 = np.array([])
        for i in xrange(0,bbbr):
            BBB2 = np.append(BBB2, int(0*BBB[i,0] + 1*BBB[i,1] + 2*BBB[i,2]))
        for i in xrange(0,nr):
            c2.execute('''INSERT INTO WebApp(Prediction, Title, Review, Summary, Rating, nR_Words) VALUES (?,?,?,?,?,?)''', (BBB2[i], AT[i,0], AT[i,1], AT[i,2], A[i,0], A[i,1], ))
            conn2.commit()
        
    
if __name__ == '__main__':
  main()

#Reviews(Usefulness, nGReview, Title, Rating, nTReview, nRWords, nRSWords, nRFS, nRVerb, nRUVerb, nRAdj, nRUAdj, nRAdv, nRUAdv, nRSVerb, nRSUVerb, nRSAdj, nRSUAdj,nRSAdv, nRSUAdv)
#(1.0, 2.0, u'The Solar Home Book: Heating, Cooling and Designing With the Sun', 5.0, 2.0, 36, 1, 4.0, 8, 7, 13, 11, 8, 6, 0, 0, 0, 0, 0, 0)

#r = 'If people become the books they read and if "the child is father to the man," then Dr. Seuss (Theodor Seuss Geisel) is the most influential author, poet, and artist of modern times. For me, a daddy to a large family who learned to read with Dr. Seuss and who has memorized too many of the books via repeated readings to young children, Prof. Nel's brilliant 'American Icon' is a long awaited treat. At last a serious treatment of this remarkable genius that is both an engaging read and filled with remarkable insights! I especially enjoyed (and learned more than I care to admit from) Prof. Nel's discussions of the Disneyfication of Seuss - which Nel links to failings in American copyright law, "the other sides of Dr. Seuss" - all of which sides were new to me, and the political genesis of his secular morality in the WWII cartoon work he did at PM magazine. The chapters on Geisel's poetry and artwork and the link Nel makes between Seuss and the historical avant guarde alone make this book a "must buy" for parents and serious readers, not to mention public libraries. Readers of Nel's other books will find the same engaging writing style that makes the book a fun read while imparting a mountain of information and important ideas. This is simply the best and most comprehensive book yet written on the work of Seuss Geisel and what will certainly be the standard for many years to come. Thank you, Prof. Nel, wherever you are, from a reader who grew up with the good doctor and who is growing up with him again years later. Your book, written from your encyclopeadic knowledge of children's literature and the media of this genre - from scanning verse to cubist painting! - explains the power, limits, and popularity of the Seuss phenomenon.'
'''
Rating

Length
adjectives
adverbs
verbs
nouns
'because'
'like'
'.'
'I'
','
'''

