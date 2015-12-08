# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:25:30 2015
Convert GBK files to UTF-8 using 'iconv' tool(of course only for linux users).
@author: hehe
"""
import os
import os.path
import subprocess
for (dirname, dirs, files) in os.walk('.'):
    for file in files:
        if file.endswith('.txt'):
            gotfile = os.path.join(dirname, file)
            #os.system(" ".join(["iconv", "-f gbk -t utf8//IGNORE","\""+gotfile+"\"", "-o", "\""+gotfile+"\""]))
            os.system(" ".join(["head -n1", "\""+gotfile+"\""]))