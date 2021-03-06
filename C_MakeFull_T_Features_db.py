"""
Script to create the features and store it into an SQL database
This will be read by the classification model
"""
import timeit, nltk
from textblob import TextBlob
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
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from collections import Counter
from nltk.stem.porter import PorterStemmer

#CREATE TABLE IF NOT EXISTS Reviews(Usefulness, R_rating, nGReview, Title, Rating, nTReview, ReviewSummary, ReviewText)

def getFeatures(t):
    porter_stemmer = PorterStemmer()
    nst = t
    nFS = 0.0
    nV = 0.0
    nN = 0.0
    nAdj = 0.0
    nAdv = 0.0
    nC = 0.0

    # Sentiment analysis
    tPOL = TextBlob(t).sentiment.polarity 
    tSUB = TextBlob(t).sentiment.subjectivity
    
    # Part-of-speech tagging
    t1 = nltk.word_tokenize(t)
    st = []
    for w in t1:
        st.append(porter_stemmer.stem(w))
    t2 = nltk.pos_tag(t1)
    nW = len(t2)

    # Lexical diversity
    if len(t2)!=0:
        nLD = 1.0*len(set(t2))/len(t2)
        nLD2 = 1.0*len(set(st))/len(st)
    else:
        nLD = 0.0
        nLD2 = 0.0

    counts = Counter(tag for word,tag in t2)
    genres = ['classic','comic','graphic novel','crime','detective','fable','fairy tale','fanfiction','fan fiction','fantasy','folklore','horror','humor','humour','legend','realism','mystery','mythology','science fiction','sci-fi','suspense','thriller','biography','autobiography','essay','textbook','history','philosophy','art','erotica','mystery','science','poetry']
    ctrG = 0
    for g in genres:
        if g in t:
            ctrG = ctrG + 1
            break
    
    # Number of sentences using sentence tokenizer
    nFS = float(len(nltk.sent_tokenize(t)))

    # Counting the number of commas, nouns, adjectives and adverbs
    for c in counts.items():
        if c[0]==',':
            nC = c[1]
        elif c[0]=='NN' or c[0]=='NNP' or c[0]=='NNS' or c[0]=='NNPS':
            nN = nN + c[1]
        elif c[0]=='JJ' or c[0]=='JJR' or c[0]=='JJS':
            nAdj = nAdj + c[1]
        elif c[0]=='VB' or c[0]=='VBD' or c[0]=='VBG' or c[0]=='VBN' or c[0]=='VBP' or c[0]=='VBZ':
            nV = nV + c[1]
        elif c[0]=='RB' or c[0]=='RBR' or c[0]=='RBS':
            nAdv = nAdv + c[1]
    if ctrG>0:
        nG = 1
    else:
        nG = 0

    # Removing stop words
    nst = re.sub("[^a-zA-Z]", " ", nst)
    nst = nst.split()
    nst = [w for w in nst if not w in stopwords.words("english")]
    nNSW = float(len(nst))
    
    # Returning all the features
    if nFS==0.0:
        nFS = 1.0
    if nW==0:
        return nW, "%.2f"%(float(nNSW/nFS)), 0.0 , tPOL, tSUB, nLD, nLD2, nFS, nC, nN/nFS, nV/nFS, nAdj/nFS, nAdv/nFS, nG
    return nW, "%.2f"%(float(nNSW/nFS)), "%.2f"%(float(nNSW/nW)), tPOL, tSUB, nLD, nLD2, nFS, nC, nN/nFS, nV/nFS, nAdj/nFS, nAdv/nFS, nG

def makeFeaturesDB(N, T1, T2, DB_file):
    print 'Go!'
    conn = sqlite3.connect(DB_file)
    c = conn.cursor()
    start_time = timeit.default_timer()
    n = 0
    nG = 0
    nB = 0
    nM = 0
    F_file = 'Chase/TT_Features'+DB_file[DB_file.find('_minR'):]

    # Creating the feature SQL table
    conn1 = sqlite3.connect(F_file)
    c1 = conn1.cursor()
    c1.execute('''DROP TABLE Features''')
    c1.execute('''CREATE TABLE IF NOT EXISTS Features(Usefulness, ReviewText, SummaryText, Rating, nR_Words, nR_NSWords1, nR_NSWords2, R_POL, R_SUB, nR_LD, nR_LD2, nR_FS, nR_C, nR_N, nR_V, nR_Adj, nR_Adv, nR_G, nS_Words, S_POL, S_SUB, nS_LD, nS_FS, nS_C, nS_N, nS_V, nS_Adj, nS_Adv)''')
    start_time = timeit.default_timer()
    for row in c.execute("SELECT * FROM Reviews"):
        if N>0 and nG>N/3 and nB>N/3 and nM>N/3:
            print nG, '  ', nB, '  ', nM,'   ',timeit.default_timer()-start_time
            conn1.close()
            return
        if (row[0]>T2 and nG<=N/3) or (row[0]<T1 and nB<=N/3) or (row[0]>=T1 and row[0]<=T2 and nM<=N/3):
            r = row[7]
            s = row[6]
            nR_Words, nR_NSWords1, nR_NSWords2, R_POL, R_SUB, nR_LD, nR_LD2, nR_FS, nR_C, nR_N, nR_V, nR_Adj, nR_Adv, nR_G = getFeatures(r)
            nS_Words, nS_NSWords1, nS_NSWords2, S_POL, S_SUB, nS_LD, nS_LD2, nS_FS, nS_C, nS_N, nS_V, nS_Adj, nS_Adv, nS_G = getFeatures(s)
            if (nG+nB+nM)%1000==0:
                print nG, '  ', nB, '  ', nM,'   ',(nG+nB+nM), '  ', timeit.default_timer()-start_time
            c1.execute('''INSERT INTO Features(Usefulness, ReviewText, SummaryText, Rating, nR_Words, nR_NSWords1, nR_NSWords2, R_POL, R_SUB, nR_LD, nR_LD2, nR_FS, nR_C, nR_N, nR_V, nR_Adj, nR_Adv, nR_G, nS_Words, S_POL, S_SUB, nS_LD, nS_FS, nS_C, nS_N, nS_V, nS_Adj, nS_Adv) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (row[0], row[7], row[6], row[4], nR_Words, nR_NSWords1, nR_NSWords2, R_POL, R_SUB, nR_LD, nR_LD2, nR_FS, nR_C, nR_N, nR_V, nR_Adj, nR_Adv, nR_G, nS_Words, S_POL, S_SUB, nS_LD, nS_FS, nS_C, nS_N, nS_V, nS_Adj, nS_Adv, ))
            conn1.commit()
            if row[0]>T2:
                nG = nG + 1
            elif row[0]<T1:
                nB = nB + 1
            else:
                nM = nM + 1
   
def main():
    minR = 20
    T1 = 0.2
    T2 = 0.8
    fT1 = T1*100
    fT2 = T2*100
    N = 1200
    nG = 0
    nB = 0
    nM = 0
    DB_file = 'Chase/T_Reviews_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db'
    makeFeaturesDB(N, T1, T2, DB_file)
    conn = sqlite3.connect('Chase/TT_Features_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
    c = conn.cursor()
    n = 0
    for row in c.execute("SELECT * FROM Features"):
        '''
        if n<100:
            print row[0], '  ', row[4], '  ', row[5], '  ', row[6], '  ', row[10], '  ', row[12], '  ', row[13], '  ', row[14], '  ', row[15], '  '
        else:
            break
        '''
        if row[0]>T2:
            nG = nG + 1
        elif row[0]<T1:
            nB = nB + 1
        else:
            nM = nM + 1
    print nG, '  ', nB, '  ', nM

if __name__ == '__main__':
  main()
