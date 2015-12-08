# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:19:41 2015

@author: hehe
"""

class TextClassify:
    def __init__(self):
        pass
    def text_classify(self, file, bow_model, classify_model):
        feature = bow_model.trainsorm_single_file(file)
        pred = classify_model.predict(feature)
        return pred
        