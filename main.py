from __future__ import division
import pandas as pd
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn import cross_validation
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from numpy.random import randint
import logging
import sys
import operator
import math
from sklearn.multiclass import OneVsRestClassifier
from collections import *

matches_info = pd.DataFrame.from_csv('selected_team_matches.csv')
#print matches_info

tournament = []
radiant = []
dire = []
winner = []
match_id = []

data = []
target = []

for i in range(0, len(matches_info)):
    tournament.append(matches_info['tournament'][i])
    radiant.append(matches_info['radiant'][i])
    dire.append(matches_info['dire'][i])
    winner.append(matches_info['winner'][i])
    match_id.append(matches_info['match_id'][i])

for i in range(0, len(matches_info)):
    #data.append([radiant[i], dire[i]])
    data.append("".join((radiant[i], " ", dire[i])))
    if winner == 'radiant':
        target.append(radiant[i])
    else:
        target.append(dire[i])

#for i in range(0, len(matches_info)):
    #print "plays {0} win {1}".format(data[i], target[i])

#now we have data and target set
#we need divide it to train and test set, do cross-validation for fitting model

print len(data), len(target)
print "data:", data[:5], "target:", target[:5]

train_data = data[:3000]
test_data = data[3000:]

train_target = target[:3000]
test_target = target[3000:]

text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC())
    ])

text_clf.fit(train_data, train_target)
print "fitted"
print "test data:", test_data
predicted = text_clf.predict(test_data)
print "predicted"
cur_score = f1_score(test_target, predicted, average='micro')
print "(", len(test_target), ", ", cur_score, ")"

#data_train = data[:]

#clf = LinearSVC()
#scores = cross_validation.cross_val_score(clf, data, target)
#print scores

#print tournament

#print len(matches_info)
#for str in matches_info:
#    print str
#    break