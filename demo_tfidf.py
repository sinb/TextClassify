# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:26:59 2015

@author: hehe
"""
import os
import numpy as np
from sklearn import linear_model

from TextClassify import TFIDFforFiles
from TextClassify import TextClassify

input_path = 'data'

train_tfidf = TFIDFforFiles(os.path.join(input_path, 'train'))
train_tf, train_df = train_tfidf.compute_tfidf()
train_tf, train_df = train_tfidf.reduce_tfidf(train_tf, train_df)

test_tfidf = TFIDFforFiles(os.path.join(input_path, 'test'))
test_tf, test_df = test_tfidf.compute_tfidf()
test_tf, _ = test_tfidf.reduce_tfidf(test_tf, test_df)

N = train_tfidf.doc_num

train_feature, train_target = train_tfidf.tfidf_feature(os.path.join(input_path, 'train'),train_tf, train_df, N)
test_feature, test_target = test_tfidf.tfidf_feature(os.path.join(input_path, 'test'),test_tf, train_df, N)

## TRAIN LR MODEL
logreg = linear_model.LogisticRegression()
logreg.fit(train_feature, train_target)

## PREDICT
test_predict = logreg.predict(test_feature)

## ACCURACY
true_false = (test_predict==test_target)
accuracy = np.count_nonzero(true_false)/float(len(test_target))
print "accuracy is %f" % accuracy

