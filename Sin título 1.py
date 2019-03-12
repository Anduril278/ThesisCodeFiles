# -*- coding: utf-8 -*-
"""
Created on Wed May 09 22:49:30 2018

@author: ErickSalvador
"""
import numpy as np
f = open ( 'myFile.txt' , 'r')
from sklearn.svm import SVC

fs1 = open ( 'fs1.txt' , 'r')
fs2 = open ( 'fs2.txt' , 'r')
fs3 = open ( 'fs3.txt' , 'r')
fs4 = open ( 'fs4.txt' , 'r')
fs5 = open ( 'fs5.txt' , 'r')
fs6 = open ( 'fs6.txt' , 'r')
fs7 = open ( 'fs7.txt' , 'r')
fs8 = open ( 'fs8.txt' , 'r')


fs1l = [[float(num) for num in line.split(',')] for line in fs1 ]
fs1l=np.array(fs1l)

fs2l = [[float(num) for num in line.split(',')] for line in fs2 ]
fs2l=np.array(fs2l)

fs3l = [[float(num) for num in line.split(',')] for line in fs3 ]
fs3l=np.array(fs3l)

fs4l = [[float(num) for num in line.split(',')] for line in fs4 ]
fs4l=np.array(fs1l)

fs5l = [[float(num) for num in line.split(',')] for line in fs5 ]
fs5l=np.array(fs5l)

fs6l = [[float(num) for num in line.split(',')] for line in fs6 ]
fs6l=np.array(fs6l)

fs7l = [[float(num) for num in line.split(',')] for line in fs7 ]
fs7l=np.array(fs7l)

fs8l = [[float(num) for num in line.split(',')] for line in fs8 ]
fs8l=np.array(fs1l)

X = np.array([fs1l,fs2l,fs3l,fs4l,fs5l,fs6l,fs7l,fs8l])
y = np.array([1, 2, 3, 3, 4, 4, 5, 5])


