from ABC import CHA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from FeatureRanking import FeatureRanking

RUN_FEATURE_RANKING = 0

if __name__ == "__main__":
    if RUN_FEATURE_RANKING:
        #1. use xgboost do feature ranking
        ds_train = "data/human-activity-recognition-with-smartphones/train.csv"
        ds_test = "data/human-activity-recognition-with-smartphones/test.csv"
        fr = FeatureRanking(ds_train,ds_test)
        fr.loadData()
        fr.featureRanking()

    #2. use abc to do feature selection
    path_train_data = "data/train_ranking.csv"
    path_test_data = "data/test_ranking.csv"
    path_fscore = 'data/Fscore_ranked.csv'
    path_name = 'data/name_ranked.csv'

    header_names = pd.read_csv(path_name).as_matrix().ravel()
    fscores =  pd.read_csv(path_fscore).as_matrix().ravel().astype(float)
    sz = len(fscores)
    fscores_minmax = np.reshape(fscores, (sz, 1))
    scaler = MinMaxScaler()
    scaler.fit(fscores_minmax)
    fscore_mean = scaler.transform(fscores_minmax)
    fscore_mean = np.reshape(fscore_mean,sz)

    cha = CHA(path_train_data,path_test_data, fscore_mean,header_names)
    cha.runCHA()