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



def checkmetrics(pred, labels_test, name):
    sns.set()
    print('The accuracy of ', name, 'is: ', accuracy_score(pred, labels_test))
    matrix = confusion_matrix(labels_test, pred)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    print(ax)
    print(classification_report(pred, labels_test))


num_classes = 6

train_df = pd.read_csv("data/human-activity-recognition-with-smartphones/train.csv")
test_df = pd.read_csv("data/human-activity-recognition-with-smartphones/test.csv")


shape = train_df.shape
rows = shape[0]
cols = shape[1]
print("Training Dataset")
print("================")
print("Columns:", shape[1])
print("Rows   :", shape[0])


target_train_df = train_df.filter(items=["Activity"])
target_test_df = test_df.filter(items=["Activity"])

target_train_df['Activity'] = pd.Categorical(target_train_df.Activity)
target_test_df['Activity'] = pd.Categorical(target_test_df.Activity)

target_train_cat_df = target_train_df.copy()
target_test_cat_df = target_test_df.copy()

target_train_cat_df['Code'] = target_train_cat_df['Activity'].cat.codes
target_test_cat_df['Code'] = target_test_cat_df['Activity'].cat.codes

features_train_df = train_df.drop("Activity", axis=1)
features_test_df = test_df.drop("Activity", axis=1)

target_train_cat_df = target_train_cat_df.drop("Activity", axis=1)
target_test_cat_df = target_test_cat_df.drop("Activity", axis=1)

features_train = features_train_df.as_matrix()
features_test = features_test_df.as_matrix()


target_test = target_test_df.as_matrix().ravel()
target_train = target_train_df.as_matrix().ravel()

target_train_cat = target_train_cat_df.as_matrix().ravel()
target_test_cat = target_test_cat_df.as_matrix().ravel()

train_processed_df = pd.concat([features_train_df, target_train_df, target_train_cat_df], axis=1)
train_processed_df.to_csv("data/train_processed.csv")

test_processed_df = pd.concat([features_test_df, target_test_df, target_test_cat_df], axis=1)
test_processed_df.to_csv("data/test_processed.csv")


shape = features_train.shape
print("Training Dataset")
print("================")
print("No. of features:", shape[1])

start_time = time.time()

xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'lambda': 0.8,
    'alpha': 0.4,
    'silent': 1,
    'num_class': num_classes
}

dtrain = xgb.DMatrix(features_train_df, target_train_cat_df)
dtest = xgb.DMatrix(features_test_df, target_test_cat_df)

num_boost_rounds = 250

watchlist = [(dtrain, 'train'), (dtest, 'test')]

# train model
xgb_model = xgb.train(xgb_params, dtrain, num_boost_rounds, watchlist)

print("Running for %s seconds" % (time.time() - start_time))


start_time = time.time()

importance = xgb_model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

def feature_selection(dataframe, importance_scores_df, threshold=0.4):
    normalized_df = importance_scores_df.copy()
    normalized_df['fscore'] = (importance_scores_df['fscore'] - importance_scores_df['fscore'].min())/(importance_scores_df['fscore'].max()-importance_scores_df['fscore'].min())
    normalized_df = normalized_df[normalized_df['fscore'] >= threshold]
    new_dataframe = dataframe.filter(items=normalized_df['feature'].tolist())
    return new_dataframe

importance_df = pd.DataFrame(importance, columns=['feature', 'fscore'])
importance_df =  importance_df.sort_values(by=['fscore'],ascending=False)


ar = np.zeros((rows,cols))
d = {}
sz = len(importance)
for i in range(0,sz):
    print(importance_df['fscore'][sz-i-1])
    ar[:,i] = train_df[importance[sz-i-1][0]]


cols_name = []
fscore = []

for i in range(0,sz):
    cols_name.append(importance[sz-i-1][0])
    fscore.append(importance_df['fscore'][sz-i-1])
print('fscore: --------------------------')
print(fscore)
print('Column names: ---------------------------')
print(cols_name)
cols_name = np.array(cols_name).flatten()
data_ranking_df = pd.DataFrame(ar)
data_ranking_df.to_csv('data/train_ranking.csv',  index=False)


print("end")