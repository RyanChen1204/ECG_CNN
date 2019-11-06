# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:23:38 2019

@author: HP
"""
import sklearn
from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
x,y=mnist['data'],mnist['target']
print(x.shape)
print(y.shape)