import pandas as pd
import numpy as np
import seaborn as sns
import time

from scipy import stats, integrate
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.stats import reciprocal, uniform

import pylab as pl
from itertools import cycle
from sklearn import cross_validation
from sklearn.svm import SVC
import xgboost as xgb
import operator

import csv

NUM_CLASS = 6

xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'lambda': 0.8,
    'alpha': 0.4,
    'silent': 1,
    'num_class': NUM_CLASS
}

class FeatureRanking:
    def __init__(self,csv_train,csv_test):
        self.csv_train = csv_train
        self.csv_test = csv_test
        self.num_boost_rounds = 250


    def checkmetrics(pred, labels_test, name):
        sns.set()
        print('The accuracy of ', name, 'is: ', accuracy_score(pred, labels_test))
        matrix = confusion_matrix(labels_test, pred)
        ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        print(ax)
        print(classification_report(pred, labels_test))
