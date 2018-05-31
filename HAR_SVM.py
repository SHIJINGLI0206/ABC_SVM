# coding: utf-8

# In[1]:

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


# In[2]:


def checkmetrics(pred, labels_test, name):
    sns.set()
    print('The accuracy of ', name, 'is: ', accuracy_score(pred, labels_test))
    matrix = confusion_matrix(labels_test, pred)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    print(ax)
    print(classification_report(pred, labels_test))


# There are six (6) classes of target variable
# 
# * LAYING
# * SITTING 
# * STANDING
# * WALKING
# * WALKING_DOWNSTAIRS
# * WALKING_UPSTAIRS

# In[3]:


num_classes = 6


# In[4]:


train_df = pd.read_csv("data/human-activity-recognition-with-smartphones/train.csv")
test_df = pd.read_csv("data/human-activity-recognition-with-smartphones/test.csv")


# In[5]:


shape = train_df.shape
print("Training Dataset")
print("================")
print("Columns:", shape[1])
print("Rows   :", shape[0])


# In[6]:


shape = test_df.shape
print("Test Dataset")
print("================")
print("Columns:", shape[1])
print("Rows   :", shape[0])


# In[7]:


train_df.head(5)


# In[8]:


### Extract features and labels from dataset for local testing:
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


# In[9]:


train_processed_df = pd.concat([features_train_df, target_train_df, target_train_cat_df], axis=1)
train_processed_df.to_csv("data/train_processed.csv")

test_processed_df = pd.concat([features_test_df, target_test_df, target_test_cat_df], axis=1)
test_processed_df.to_csv("data/test_processed.csv")


# In[10]:


shape = features_train.shape
print("Training Dataset")
print("================")
print("No. of features:", shape[1])


# # Feature Selection using XGBoost Feature Importance

# In[13]:


import xgboost as xgb
import operator

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

num_boost_rounds = 5

watchlist = [(dtrain, 'train'), (dtest, 'test')]

# train model
xgb_model = xgb.train(xgb_params, dtrain, num_boost_rounds, watchlist)

print("Running for %s seconds" % (time.time() - start_time))


# In[14]:


start_time = time.time()

importance = xgb_model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

importance_df = pd.DataFrame(importance, columns=['feature', 'fscore'])

# Plot Feature Importance
plt.figure()
importance_df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))

print("Running for %s seconds" % (time.time() - start_time))


# In[15]:


def feature_selection(dataframe, importance_scores_df, threshold=0.4):
    normalized_df = importance_scores_df.copy()
    normalized_df['fscore'] = (importance_scores_df['fscore'] - importance_scores_df['fscore'].min())/(importance_scores_df['fscore'].max()-importance_scores_df['fscore'].min())
    normalized_df = normalized_df[normalized_df['fscore'] >= threshold]
    new_dataframe = dataframe.filter(items=normalized_df['feature'].tolist())
    return new_dataframe


# In[16]:


def top_features(importance_scores_df, top=20):
    df = importance_scores_df.sort_values(by=['fscore'], ascending=False)
    return df.head(top)


# In[23]:


start_time = time.time()

top20_df = top_features(importance_df, top=20)
top20_df = top20_df.sort_values(by=['fscore'], ascending=True)

# Plot Feature Importance
plt.figure()
top20_df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))

print("Running for %s seconds" % (time.time() - start_time))


# ## Keep impotant features within 1 standard deviation

# In[ ]:

### -----------

# ## Keep impotant features within 2 standard deviation

# In[34]:


# two standard deviation away from the best feature
threshold_2std = 1 - 0.9545

features_train_2std_df = feature_selection(features_train_df, importance_df, threshold=threshold_2std)
features_test_2std_df = feature_selection(features_test_df, importance_df, threshold=threshold_2std)

features_train_2std_df.to_csv("data/features_train_2std.csv", index=False)
features_test_2std_df.to_csv("data/features_test_2std.csv", index=False)


# In[13]:


features_train_2std_df = pd.read_csv("data/features_train_2std.csv")
features_test_2std_df = pd.read_csv("data/features_test_2std.csv")

features_train_2std = features_train_2std_df.as_matrix()
features_test_2std = features_test_2std_df.as_matrix()
print('features train 2std: ',features_train_2std[:][0])


# In[14]:


shape = features_train_2std.shape
print("Training Dataset (2 standard deviation)")
print("================")
print("No. of features:", shape[1])


# ## C-Support Vector Classification with XGBoost 2 sd (165 features)

# In[37]:


# ## Linear Support Vector Classification with XGBoost 2 sd (165 features)

# In[38]:


start_time = time.time()

parameters ={}
SVM = LinearSVC()
grid_search_cv = GridSearchCV(SVM, parameters, cv=3,n_jobs=-1, return_train_score=True, refit=True,verbose=1)
grid_search_cv.fit(features_train_2std, target_train)
resultsdf=pd.DataFrame(grid_search_cv.cv_results_)
print("The train score:", str(grid_search_cv.score(features_train_2std, target_train)), "with parameters:", grid_search_cv.best_params_)
pred = grid_search_cv.best_estimator_.predict(features_test_2std)



checkmetrics(pred, target_test, 'Linear Support Vector Classification')

print("Running for %s seconds" % (time.time() - start_time))


