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
        self.features_train_df = None
        self.target_train_cat_df = None
        self.features_test_df = None
        self.target_test_cat_df = None
        self.train_df = None
        self.test_df = None
        self.target_train_cat = None
        self.target_test_cat = None

        self.rows_train = 0
        self.rows_test = 0


    def checkmetrics(pred, labels_test, name):
        sns.set()
        print('The accuracy of ', name, 'is: ', accuracy_score(pred, labels_test))
        matrix = confusion_matrix(labels_test, pred)
        ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        print(ax)
        print(classification_report(pred, labels_test))

    def loadData(self):
        print('FeatureRanking: load data.')
        self.train_df = pd.read_csv(self.csv_train)
        self.test_df = pd.read_csv(self.csv_test)

        self.rows_train,cols_train = self.train_df.shape
        self.rows_test, cols_test = self.test_df.shape

        print("Training Dataset")
        print("================")
        print("Columns:", cols_train)
        print("Rows   :", self.rows_train)
        print("Test Dataset")
        print("================")
        print("Columns:", cols_test)
        print("Rows   :", self.rows_test)

        target_train_df = self.train_df.filter(items=["Activity"])
        target_test_df = self.test_df.filter(items=["Activity"])

        target_train_df['Activity'] = pd.Categorical(target_train_df.Activity)
        target_test_df['Activity'] = pd.Categorical(target_test_df.Activity)

        self.target_train_cat_df = target_train_df.copy()
        self.target_test_cat_df = target_test_df.copy()

        self.target_train_cat_df['Code'] = self.target_train_cat_df['Activity'].cat.codes
        self.target_test_cat_df['Code'] = self.target_test_cat_df['Activity'].cat.codes

        self.features_train_df = self.train_df.drop("Activity", axis=1)
        self.features_test_df = self.test_df.drop("Activity", axis=1)

        self.target_train_cat_df = self.target_train_cat_df.drop("Activity", axis=1)
        self.target_test_cat_df = self.target_test_cat_df.drop("Activity", axis=1)

        features_train = self.features_train_df.as_matrix()
        features_test = self.features_test_df.as_matrix()

        target_test = target_test_df.as_matrix().ravel()
        target_train = target_train_df.as_matrix().ravel()

        self.target_train_cat = self.target_train_cat_df.as_matrix().ravel()
        self.target_test_cat = self.target_test_cat_df.as_matrix().ravel()

        train_processed_df = pd.concat([self.features_train_df, target_train_df, self.target_train_cat_df], axis=1)

        test_processed_df = pd.concat([self.features_test_df, target_test_df, self.target_test_cat_df], axis=1)


    def feature_selection(dataframe, importance_scores_df, threshold=0.4):
        normalized_df = importance_scores_df.copy()
        normalized_df['fscore'] = (importance_scores_df['fscore'] - importance_scores_df['fscore'].min()) / (
        importance_scores_df['fscore'].max() - importance_scores_df['fscore'].min())
        normalized_df = normalized_df[normalized_df['fscore'] >= threshold]
        new_dataframe = dataframe.filter(items=normalized_df['feature'].tolist())
        return new_dataframe

    def featureRanking(self):
        print('FeatureRanking: featureRanking')
        dtrain = xgb.DMatrix(self.features_train_df, self.target_train_cat_df)
        dtest = xgb.DMatrix(self.features_test_df, self.target_test_cat_df)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]

        # train model
        xgb_model = xgb.train(xgb_params, dtrain, self.num_boost_rounds, watchlist)

        importance = xgb_model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        importance_df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        importance_df = importance_df.sort_values(by=['fscore'], ascending=False)

        sz = len(importance)
        ar_train = np.zeros((self.rows_train, sz))
        ar_test = np.zeros((self.rows_test, sz))


        for i in range(0, sz):
            ar_train[:, i] = self.train_df[importance[sz - i - 1][0]]
            ar_test[:, i] = self.test_df[importance[sz - i - 1][0]]
        ar_train = pd.DataFrame(ar_train)
        ar_test = pd.DataFrame(ar_test)
        ar_train = ar_train.assign(score=self.target_train_cat)
        ar_test = ar_test.assign(score=self.target_test_cat)

        cols_name = []
        fscore = []

        # write ranking data into csv file
        for i in range(0, sz):
            cols_name.append(importance[sz - i - 1][0])
            fscore.append(importance_df['fscore'][sz - i - 1])

        f = pd.DataFrame(fscore)
        f.to_csv('data/Fscore_ranked.csv',index=False)
        c = pd.DataFrame(cols_name)
        c.to_csv('data/name_ranked.csv',index=False)

        s = ['score']
        cols_name.append('score')
        ar_train.columns = cols_name

        ar_train.to_csv('data/train_ranking.csv', index=False)
        ar_test.to_csv('data/test_ranking.csv', index=False)
        print('FeatureRanking: Done!')


# if __name__ == '__main__':
#     print('Start Feature Ranking....')
#     ds_train = "data/human-activity-recognition-with-smartphones/train.csv"
#     ds_test = "data/human-activity-recognition-with-smartphones/test.csv"
#     fr = FeatureRanking(ds_train,ds_test)
#     fr.loadData()
#     fr.featureRanking()
#     print('Done!')

