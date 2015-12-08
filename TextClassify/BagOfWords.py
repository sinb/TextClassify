# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:25:58 2015

@author: hehe
"""

import os
import re
import jieba
import numpy
from scipy import sparse

class BagOfWords:
    def __init__(self, dir):
        self.dir = dir
        
    def build_dictionary(self):
        dict_set = set()        
        pattern1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        pattern2 = re.compile('保存时间:.*')        
        count = 0
        for (dirname, dirs, files) in os.walk(self.dir):
            for file in files:
                if file.endswith('.txt'):
                    filename = os.path.join(dirname, file)
                    with open(filename, 'rb') as f:
                        count += 1
                        for line in f:
                            m1 = pattern1.findall(line)
                            m2 = pattern2.findall(line)
#                            if m1 or m2:
#                                continue
                            line = self.process_line(line)
                            words = jieba.cut(line.strip(), cut_all=False)
                            dict_set |= set(words)
        self.num_samples = count
        self.dict = self.reduce_dict(dict_set)
        
    def load_dictionary(self, dir):
        import cPickle as Pickle  
        try:
            print "loaded dictionary from %s" % dir
            self.dict = Pickle.load(open(dir, 'rb'))
            print "done"            
        except IOError:
            print "error while loading from %s" % dir
            
    def save_dictionary(self, dir):
        import cPickle as Pickle
        Pickle.dump(self.dict, open(dir, 'wb'))
        print "saved dictionary to %s" % dir
                
    def reduce_dict(self, dict_set):
        dict_copy = dict_set.copy()
        for word in dict_set:
            if len(word) < 2:
                dict_copy.remove(word)
            else:
                try:
                    float(word)
                    dict_copy.remove(word)
                except ValueError:
                    continue
        dictionary = {}
        for idx, word in enumerate(dict_copy):
            dictionary[word] = idx
        return dictionary

    def process_line(self, line):
        line = line.decode("utf8")
        return re.sub("]-·[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）():\"=《]+".decode("utf8"),
                                           " ".decode("utf8"), line)      

    def transform_data(self, dir):
        from scipy import sparse
        print "transforming data in to bag of words vector"        
        data = []
        target = []
        pattern1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        pattern2 = re.compile('保存时间:.*')      
        count = 0
        for (dirname, dirs, files) in os.walk(dir):
            for file in files:
                if file.endswith('.txt'):
                    count += 1
                    filename = os.path.join(dirname, file)
                    tags = re.split('[/\\\\]', dirname)
                    tag = tags[-1]
                    word_vector = numpy.zeros(len(self.dict))
                    with open(filename, 'rb') as f:
                        for line in f:
                            m1 = pattern1.findall(line)
                            m2 = pattern2.findall(line)
#                            if m1 or m2:
#                                continue
                            line = self.process_line(line)
                            words = jieba.cut(line.strip(), cut_all=False)
                            for word in words:
                                try:
                                    word_vector[self.dict[word]] += 1
                                except KeyError:
                                    pass
                    #data.append(sparse.csr_matrix(word_vector))
                    data.append(word_vector)
                    target.append(tag)
        self.num_samples = count
        print "done"                            
        return sparse.csr_matrix(numpy.asarray(data)),numpy.asarray(target)
    
    def trainsorm_single_file(self, file):
        pattern1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        pattern2 = re.compile('保存时间:.*')   
        word_vector = numpy.zeros(len(self.dict))
        with open(file, 'rb') as f:
            for line in f:
                m1 = pattern1.findall(line)
                m2 = pattern2.findall(line)
#                if m1 or m2:
#                    continue
                line = self.process_line(line)
                words = jieba.cut(line.strip(), cut_all=False)
                for word in words:
                    try:
                        word_vector[self.dict[word]] += 1
                    except KeyError:
                        pass
        return word_vector