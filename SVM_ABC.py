from ABC import CHA
import numpy as np
from sklearn.preprocessing import MinMaxScaler





if __name__ == "__main__":
    #path_data = "data/human-activity-recognition-with-smartphones/train.csv"
    path_data = "data/train_ranking.csv"
    f = open('data/Fscore_ranked', 'r')
    s = f.read()
    l = s.split(',')
    fscores = np.array(l,dtype=float)
    f.close()


    sz = len(fscores)
    fscores_minmax = np.reshape(fscores, (sz, 1))
    scaler = MinMaxScaler()
    scaler.fit(fscores_minmax)
    fscore_mean = scaler.transform(fscores_minmax)
    fscore_mean = np.reshape(fscore_mean,sz)



    f = open('data/name_ranked')
    s = f.read()
    l = s.split(',')
    header_names = np.array(l)
    f.close()

    #only for tran_ranking.csv
    valid_col_num = 512
    target_col_index = 563


    cha = CHA(path_data, fscore_mean,header_names, valid_col_num,target_col_index)
    cha.runCHA()