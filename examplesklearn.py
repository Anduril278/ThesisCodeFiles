# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:24:49 2018

@author: ErickSalvador
"""

from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
iris_X.shape
(150, 4)