"""
Script to create an SQL database containing the reviews
It reads the text file and creates a row for each review
"""

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
import pandas as pd
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from math import sqrt

def GetInfo(trainSize, minR, T1, T2):
    """
    Store reviews into a SQL database
    No features extracted at this point
    minR - minimum of number of ratings required for the review
    T1 - threshold for not helpful
    T2 - threshold for helpful
    Populate the three classes equally
    """
    start_time = timeit.default_timer()
    fT1 = T1*100
    fT2 = T2*100
    conn = sqlite3.connect('Chase/T_Reviews_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
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
        if 'product/title' in l: # Title of tje book
            start = True
            title = l[(l.find(':'))+1:].strip()
        elif 'review/score' in l: # Rating of the book
            rating = float(l[l.find(':')+1:])
        elif 'review/helpfulness' in l: # Helpfulness of the book
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
        elif 'review/summary' in l and UD>minR: # Review summary text
            r = l[l.find(':')+1:]
            summary = l[l.find(':')+1:]
        elif 'review/text' in l and UD>minR: # Review text
            text = l[l.find(':')+1:]
            nF = nF + 1
            if start==True and UD>minR: # Only store information where at least minR people have rated the review
                if len(Title)==trainSize:
                    stop_time = timeit.default_timer()
                    conn.close()
                    return start_time,stop_time
                if (float(U)>T2 and nG<=trainSize/3) or (float(U)<T1 and nB<=trainSize/3) or (float(U)>=T1 and float(U)<=T2 and nM<=trainSize/3): # Ensure equal number in each class
                    Title.append(l[(l.find(':'))+1:])
                    c.execute('''INSERT INTO Reviews(Usefulness, R_rating, nGReview, Title, Rating, nTReview, ReviewSummary, ReviewText) VALUES (?,?,?,?,?,?,?,?)''', (float(U), R_rating, UN, title, rating, UD, summary, text, ))
                    start = False
                    conn.commit()
                    if len(Title)%5000==0:
                        print int(len(Title)/5000), '   ',timeit.default_timer()-start_time
    stop_time = timeit.default_timer()
    conn.close()
    return start_time,stop_time

def make_db_file(N,minR,T1,T2):
    start_time,stop_time =  GetInfo(N,minR,T1,T2)
    print stop_time - start_time

def main():
    N = 1200
    minR = 20
    T1 = 0.2
    T2 = 0.8
    fT1 = T1*100
    fT2 = T2*100
    make_db_file(N,minR,T1,T2)
    conn = sqlite3.connect('Chase/T_Reviews_minR'+str(minR)+'_T1_'+str(int(fT1))+'_T2_'+str(int(fT2))+'.db')
    c = conn.cursor()
    n = 0
    for row in c.execute('''SELECT * FROM Reviews'''):
        n = n + 1
    print n
    conn.close()
    
if __name__ == '__main__':
  main()
