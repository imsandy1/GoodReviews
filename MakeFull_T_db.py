#Python script to create SQL database storing the book review information
import timeit
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

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sknn.mlp import Classifier, Layer

import pandas as pd
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from math import sqrt

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

def GetInfo(trainSize, minR, T1, T2):
    start_time = timeit.default_timer()
    fT1 = T1*100
    fT2 = T2*100
    conn = sqlite3.connect('../files/T_Reviews_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
    c = conn.cursor()
    c.execute('''DROP TABLE Reviews''')
    c.execute('''CREATE TABLE IF NOT EXISTS Reviews(Usefulness, R_rating, nGReview, Title, Rating, nTReview, ReviewSummary, ReviewText)''')
    f = open('../files/Books.txt')
    ctr = 0
    Title = []
    rating = -1.0
    title = ''
    text = ''
    summary = ''
    U = -1.0
    UD = -1.0
    UN = -1.0
    nFS = -1.0
    start = False
    nG = 0
    nB = 0
    nM = 0
    R_rating = -1.0
    nF = 0
    for l in f:
        l.strip()
        if 'product/title' in l:
            start = True
            title = l[(l.find(':'))+1:].strip()
        elif 'review/score' in l:
            rating = float(l[l.find(':')+1:])
        elif 'review/helpfulness' in l:
            i1 = l.find(':')
            ll = l[i1+1:]
            i2 = ll.find('/')
            UN = float(ll[:i2])
            UD = float(ll[i2+1:])
            if UD>0:# and UD>minR
                U = "%.2f"%(UN/UD)
                R_rating = "%.2f"%((2.0*UN - UD)/(UD))
                if float(U)>T2 and UD>minR:
                    nG = nG + 1
                elif float(U)<T1 and UD>minR:
                    nB = nB + 1
                elif float(U)>=T1 and float(U)<=T2 and UD>minR:
                    nM = nM + 1
            else:
                U = 0.0
                R_rating = 0.0
        elif 'review/summary' in l and UD>minR:
            r = l[l.find(':')+1:]
            summary = l[l.find(':')+1:]
        elif 'review/text' in l and UD>minR:
            text = l[l.find(':')+1:]
            nF = nF + 1
            if start==True and UD>minR:
                if len(Title)==trainSize:
                    stop_time = timeit.default_timer()
                    conn.close()
                    return start_time,stop_time
                if (float(U)>T2 and nG<=trainSize/3) or (float(U)<T1 and nB<=trainSize/3) or (float(U)>=T1 and float(U)<=T2 and nM<=trainSize/3):
                    Title.append(l[(l.find(':'))+1:])
                    c.execute('''INSERT INTO Reviews(Usefulness, R_rating, nGReview, Title, Rating, nTReview, ReviewSummary, ReviewText) VALUES (?,?,?,?,?,?,?,?)''', (float(U), R_rating, UN, title, rating, UD, summary, text, ))
                    start = False
                    conn.commit()
                    if len(Title)%5000==0:
                        print int(len(Title)/5000), '   ',timeit.default_timer()-start_time
    stop_time = timeit.default_timer()
    conn.close()
    return start_time,stop_time
#c.execute('''CREATE TABLE IF NOT EXISTS Reviews(Usefulness, nGReview, Title, Rating, nTReview, ReviewSummary, ReviewText)''')
def make_db_file(N,minR,T1,T2):
    start_time,stop_time =  GetInfo(N,minR,T1,T2)
    print stop_time - start_time

def doFactorScaling(z):
    X = z[:]
    XX = (X - np.mean(X))/np.std(X)
    return XX

def main():
    N = 1200000
    minR = 20
    T1 = 0.2
    T2 = 0.8
    fT1 = T1*100
    fT2 = T2*100
    make_db_file(N,minR,T1,T2)
    conn = sqlite3.connect('../files/T_Reviews_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
    c = conn.cursor()
    n = 0
    for row in c.execute('''SELECT * FROM Reviews'''):
        n = n + 1
    print n
    conn.close()
    
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
