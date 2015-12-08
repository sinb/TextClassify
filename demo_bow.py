# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:36:00 2015

@author: hehe
"""
import os
import numpy as np
from sklearn import linear_model

from TextClassify import BagOfWords
from TextClassify import TextClassify

data_dir = 'data'
## BAG OF WORDS MODEL
BOW = BagOfWords(os.path.join(data_dir, 'train'))
#BOW.build_dictionary()
#BOW.save_dictionary(os.path.join(data_dir, 'dicitionary.pkl'))
BOW.load_dictionary('dicitionary.pkl')

## LOAD DATA
train_feature, train_target = BOW.transform_data(os.path.join(data_dir, 'train'))
test_feature, test_target = BOW.transform_data(os.path.join(data_dir, 'test'))

## TRAIN LR MODEL
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_feature, train_target)

## PREDICT
test_predict = logreg.predict(test_feature)

## ACCURACY

true_false = (test_predict==test_target)
accuracy = np.count_nonzero(true_false)/float(len(test_target))
print "accuracy is %f" % accuracy

## TextClassify
TextClassifier = TextClassify()
pred = TextClassifier.text_classify('test.txt', BOW, logreg)
print pred[0]