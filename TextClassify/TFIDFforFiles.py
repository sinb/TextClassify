# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:04:25 2015

@author: hehe
"""
import re
import os
from os import listdir
from os.path import isfile, join
from itertools import groupby
from operator import itemgetter
import jieba
class TFIDFforFiles:
    def __init__(self,dir):
        """
        initialize file paths
        """
        self.dirname = dir
        self.filenames = []
        for (dirname, dirs, files) in os.walk(self.dirname):
            for file in files:
                self.filenames.append(os.path.join(dirname, file))
        self.doc_num = len(self.filenames)
        
    def compute_tf_by_file(self):
        """
        for each file, calcute term frequency
        return format is :
            'word', 'file_name', term-frequency
        """
        word_docid_tf = []
        for name in self.filenames:
            with open(join(name), 'r') as f:
                tf_dict = dict()
                for line in f:
                    line = self.process_line(line)
                    words = jieba.cut(line.strip(), cut_all=False)
                    #words = line.rstrip().split(separator)
                    for word in words:
                        tf_dict[word] = tf_dict.get(word, 0) + 1
            tf_list = tf_dict.items()
            word_docid_tf += [[item[0], name, item[1]] for item in tf_list]
        return word_docid_tf      
          
    def compute_tfidf(self):
        """
        do some statistics, i.e, calcute document frequency
        """
        word_docid_tf = self.compute_tf_by_file()
        word_docid_tf.sort()
        tfidf = dict()
        doc_freq = dict()
        term_freq = dict()
        for current_word, group in groupby(word_docid_tf, itemgetter(0)):
            doclist = []
            df = 0
            for current_word, file_name, tf in group:
                doclist.append((file_name, tf))
                df += 1
            term_freq[current_word] = dict(doclist)
            doc_freq[current_word] = df
        return term_freq, doc_freq
        
    def process_line(self, line):
        try:
            line = line.decode("utf8")
            line = re.sub("]-·[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"),
                      " ".decode("utf8"),line)
            return line
        except UnicodeDecodeError:            
            return line
            
    def printFileNames(self):
        for name in self.filenames:
            print join(self.dirname, name)
            
    def reduce_tfidf(self, term_freq, doc_freq):
        remove_list = []        
        for key in term_freq.keys():
            if len(key) < 2:
                remove_list.append(key)
            else:
                try:
                    float(key)
                    remove_list.append(key)
                except ValueError:
                    continue
        for key in remove_list:
            term_freq.pop(key)
            doc_freq.pop(key)
        return term_freq, doc_freq
        
    def tfidf_feature(self, dir, term_freq, doc_freq, N):
        print "converting..."
        import numpy as np 
        import os
        from scipy import sparse
        filenames = []
        for (dirname, dirs, files) in os.walk(dir):
            for file in files:
                filenames.append(os.path.join(dirname, file))
        word_list = dict()
        for idx, word in enumerate(doc_freq.keys()):
            word_list[word] = idx
        features = []    
        target = []
        for name in filenames:
            feature = np.zeros(len(doc_freq))
            words_in_this_file = set()
            tags = re.split('[/\\\\]', name)
            tag = tags[-2]            
            with open(name, 'rb') as f:
                for line in f:
                    line = self.process_line(line)
                    words = jieba.cut(line.strip(), cut_all=False)
                    #words = line.rstrip().split(separator)
                    for word in words:
                        words_in_this_file.add(word)
            for word in words_in_this_file:       
                try:
                    idf = np.log(float(N) / doc_freq[word])
                    #idf = 1
                    tf = term_freq[word][name]
                    feature[word_list[word]] = tf*idf
                except KeyError:
                    continue
            #features.append(sparse.csr_matrix(feature))
            features.append(feature)
            target.append(tag)
        print "done"
        #return sparse.csr_matrix(features),np.asarray(target)
        return sparse.csr_matrix(np.asarray(features)), np.asarray(target)