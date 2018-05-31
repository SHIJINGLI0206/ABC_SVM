
from datetime import datetime
# from weka.classifiers import Classifier,Evaluation
# from weka.filters import Filter,MultiFilter
# from weka.core.dataset import Instances
# from weka.filters import Filter, MultiFilter, StringToWordVector
# from weka.core.dataset import Attribute, Instance
from random import randint
# from weka.core.converters import Loader,load_any_file
# import javabridge
# import weka.core.jvm as jvm
from abc import ABC, abstractmethod
from enum import Enum
from FoodSource import FoodSource
from scipy.io import arff
from io import StringIO

import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

class PerturbationStrategy(Enum):
    USE_MR = 1
    CHANGE_ONE_FEATURE = 2


class CHA():
    def __init__(self, data_train_path, data_test_path, fscores, header_names):
        #self.features = {True, True, True, True}
        # self.features = { True, True, True, True, True, True, True, True,
        #                    True, True, True, True, True, True, True, True, True, True,
        #                    True}
        self.features = 0

        self.featureSize = 0
        #self.databaseName = "dataset/segment.arff"
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.runtime =2
        self.limit = 20
        self.mr = 0.07
        self.KFOLD = 3
        self.maxNRFeatures = 19
        #fix selected number for init food source
        self.selectedFeatureNum = 25
        self.fscores = fscores
        self.header_names = header_names



        self.bestFitness = 0
        self.bestFoodSource = None
        self.foodSources = set()
        self.visitedFoodSources = set()
        self.scouts = set()
        self.abandoned = set()
        self.markedToRemoved = set()
        self.neighbors = set()
        if self.mr > 0:
            self.perturbation = PerturbationStrategy.USE_MR
        else:
            self.perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE

        self.states = 0
        self.data_train = None
        self.data_test = None

    def loadFeatures(self):
        ds_train = pd.read_csv(self.data_train_path,header=0)
        ds_test = pd.read_csv(self.data_test_path,header=0)

        #self.data = np.array(ds['data'])
        self.data_train = ds_train.as_matrix()
        self.data_test = ds_test.as_matrix()
        rows_train, cols_train = self.data_train.shape
        rows_test, cols_test = self.data_test.shape

        self.featureSize = cols_train - 1
        self.features = np.ones(self.featureSize, dtype=bool)

        return self.data_train.shape[0]


    def executeKFoldClassifier(self,featureInclusion, kFold, classIndex):
        deletedFeatures = 0
        for i in range(0,len(featureInclusion)):
            if featureInclusion[i] == False:
                self.instances.deleteAttributeAt( i - deletedFeatures)
                deletedFeatures += 1

    def initializeFoodSource(self):
        print('initializeFoodSources')
        #for i in range(0,self.featureSize):
        for i in range(0, self.selectedFeatureNum):
            self.states += 1
            features = np.zeros(self.featureSize)
            features[i] = True
            curFitness = self.calculateFitness(features)
            print('i=',i)
            fs = FoodSource(features,curFitness,1)
            self.foodSources.add(fs)
            if(curFitness >  self.bestFitness):
                self.bestFoodSource = fs
                self.bestFitness = curFitness


    def sendEmployedBees(self):
        print('sendEmployedBees')
        self.scouts = set()
        self.markedToRemoved = set()
        self.neighbors = set()
        employedbeecount = 1
        for fs in self.foodSources:
            self.sendBee(fs)
            print("sending employed bees loop no: ", employedbeecount)
            employedbeecount += 1

        # remove all markedToRemoved
        for mtr in self.markedToRemoved:
            if mtr in self.foodSources:
                self.foodSources.remove(mtr)

        for n in self.neighbors:
            self.foodSources.add(n)


    def sendOnlookerBees(self):
        print('SendOnlookerBees')
        self.markedToRemoved = set()
        self.neighbors = set()

        min = list(self.foodSources)[0].getFitness()
        range = list(self.foodSources)[0].getFitness()
        for s in self.foodSources:
            if s.getFitness() < min:
                min = s.getFitness()
            if s.getFitness() > range:
                range = s.getFitness()

        employedbeecount = 0
        for fs in self.foodSources:
            prob = (fs.getFitness()-min)/range
            r = random.random()
            if r < prob:
                self.sendBee(fs)
                print("sending onlooker bees loop no:", employedbeecount)
                employedbeecount += 1
            else:
                fs.incrementLimit()
                #print('fs limit: ', fs.getLimit())

        for mtr in self.markedToRemoved:
            if mtr in self.foodSources:
                self.foodSources.remove(mtr)

        for n in self.neighbors:
            self.foodSources.add(n)



    def sendBee(self,foodSource):
        features = foodSource.getFeatureInclusion()
        nrFeatures = foodSource.getNrFeatures()
        times = 0
        modifedFoodSource = None
        while 1:
            times += 1
            if self.perturbation == PerturbationStrategy.CHANGE_ONE_FEATURE:
                index = round(random.random() * (self.featureSize - 1))
                if features[index] is False:
                    nrFeatures += 1
                    features[index] = True
            elif self.perturbation == PerturbationStrategy.USE_MR:
                for i in range(0,self.featureSize):
                    r = random.random()
                    r = r *(1 - self.fscores[i])
                    if r < self.mr:
                        if features[i] == False and \
                                nrFeatures <= self.maxNRFeatures and \
                                np.count_nonzero(features) < self.maxNRFeatures:
                            nrFeatures += 1
                            features[i] = True

                print('*******************************')
                print('features: ', np.count_nonzero(features))
                print('*******************************')

            modifedFoodSource = FoodSource(features)
            if (modifedFoodSource not in self.foodSources and \
                            modifedFoodSource not in self.neighbors and \
                            modifedFoodSource not in self.abandoned and \
                            modifedFoodSource not in self.visitedFoodSources) or \
                            times > self.featureSize:
                break

        if modifedFoodSource not in self.foodSources or \
            modifedFoodSource not in self.neighbors or \
            modifedFoodSource not in self.visitedFoodSources or \
            modifedFoodSource not in self.abandoned:
            self.states += 1
            fitness = self.calculateFitness(features)
            modifedFoodSource.setFitness(fitness)
            modifedFoodSource.setNrFeatures(nrFeatures)
            if foodSource.getFitness() > fitness or \
                    (fitness == foodSource.getFitness() and nrFeatures > foodSource.getNrFeatures()):
                foodSource.incrementLimit()
                if foodSource.getLimit() >= self.limit:
                    self.markAbandonsFoodSource(foodSource)
                    self.createScoutBee()
                self.visitedFoodSources.add(modifedFoodSource)
            else:
                if (fitness > self.bestFitness or \
                        (fitness == self.bestFitness
                         and nrFeatures < self.bestFoodSource.getNrFeatures()
                         )) and nrFeatures <= self.maxNRFeatures:
                    print('-----------------------------------------------------')
                    num_modify = np.count_nonzero(modifedFoodSource.getFeatureInclusion())
                    print("modify feature num: ", num_modify)
                    print('-----------------------------------------------------')
                    num_best =  np.count_nonzero(self.bestFoodSource.getFeatureInclusion())
                    print('num best : ',num_best)
                    if num_modify <= self.maxNRFeatures:
                        print('++++++++++++')
                        print('update best food source.')
                        fi = modifedFoodSource.getFeatureInclusion()
                        ft = modifedFoodSource.getFitness()
                        nr = modifedFoodSource.getNrFeatures()

                        self.bestFoodSource = FoodSource(fi,ft,nr)
                        print('+++++++++++')
                        num_best_updated = np.count_nonzero(self.bestFoodSource.getFeatureInclusion())
                        print('updated best num: ',num_best_updated )
                        self.bestFitness = fitness
                self.neighbors.add(modifedFoodSource)
        return True

    def createScoutBee(self):
        features = np.zeros(self.featureSize)
        nrFeatures = 0
        #for j in range(0,self.featureSize):
        for j in range(0,self.maxNRFeatures):
            inclusio = bool(random.getrandbits(1))
            if inclusio:
                nrFeatures += 1
            features[j] = inclusio


        curFitness = self.calculateFitness(features)
        foodSource = FoodSource(features,curFitness,nrFeatures)
        if foodSource not in self.foodSources and \
                        foodSource not in self.neighbors and \
                        foodSource not in self.abandoned and \
                        foodSource not in self.visitedFoodSources:
            self.states += 1
            self.scouts.add(foodSource)

    def sendScoutBeesAndRemoveAbandonsFoodSource(self):
        #remove abandoned
        for abd in self.abandoned:
            if abd in self.foodSources:
                self.foodSources.remove(abd)

        for s in self.scouts:
            self.foodSources.add(s)

    def markAbandonsFoodSource(self,foodSource):
        self.abandoned.add(foodSource)

    def calculateFitness(self,featureInclusion):
        if 0:
            deletedFeatures = 0
            data = self.data_train
            for i in range(0,len(featureInclusion)):
                if featureInclusion[i] == False:
                    data = np.delete(data,np.s_[i-deletedFeatures],1)
                    deletedFeatures += 1

            rows, cols = data.shape
            X = data[:, :cols-1]
            y = data[:, cols-1:]
            y = y.ravel()
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            n = KNeighborsClassifier(n_neighbors=3)
            n.fit(X_train, y_train)

            score = n.score(X_test, y_test)
        else:
            deletedFeatures = 0
            data_train = self.data_train
            data_test = self.data_test
            for i in range(0, len(featureInclusion)):
                if featureInclusion[i] == False:
                    data_train = np.delete(data_train, np.s_[i - deletedFeatures], 1)
                    data_test = np.delete(data_test, np.s_[i - deletedFeatures], 1)
                    deletedFeatures += 1

            rows_train, cols_train = data_train.shape
            rows_test, cols_test = data_test.shape

            X_train = data_train[:, :cols_train - 1]
            y = data_train[:, cols_train - 1:]
            y_train = y.ravel()

            X_test = data_test[:, :cols_test - 1]
            y = data_test[:, cols_test - 1:]
            y_test = y.ravel()

            parameters = {}
            SVM = LinearSVC()
            grid_search_cv = GridSearchCV(SVM, parameters, cv=3, n_jobs=-1, return_train_score=True, refit=True,
                                          verbose=0)
            grid_search_cv.fit(X_train, y_train)
            resultsdf = pd.DataFrame(grid_search_cv.cv_results_)

            train_score = grid_search_cv.score(X_train, y_train)
            pred = grid_search_cv.best_estimator_.predict(X_test)

            score = accuracy_score(pred, y_test)
            # print("The train score:", str(score), "with parameters:",
            #    grid_search_cv.best_params_)

        return score


    def executeFeatureSelection(self):
        self.visitedFoodSources = set()
        self.states = 0
        time = datetime.now()
        self.initializeFoodSource()
        print('init time: ',datetime.now() - time)
        for i in range(0,self.runtime):
            self.sendEmployedBees()
            self.sendOnlookerBees()
            self.sendScoutBeesAndRemoveAbandonsFoodSource()

        time = (datetime.now() - time) / 60000
        self.logBestSolutionAndTime(time)
        self.states = 0

    def logBestSolutionAndTime(self,t):
        print('Time: ',t)
        print('Best ', self.bestFoodSource.getFeatureInclusion())
        featureNum = np.count_nonzero(self.bestFoodSource.getFeatureInclusion())
        print('Selected Feature Num: ',featureNum)
        print('Feature selection End.')
        print('Best Fitness is ',self.bestFitness)

    def Linear_SVM(self):
        print('Start Final Linear SVM.')
        data_train = self.data_train
        data_test = self.data_test
        fi = self.bestFoodSource.getFeatureInclusion()
        fn = np.count_nonzero(self.bestFoodSource.getFeatureInclusion())
        print('num features: ', fn)
        rows_train, cols_train = data_train.shape
        rows_test, cols_test = data_test.shape

        X_train = np.zeros((rows_train, fn),dtype=float)
        y_train = data_train[:, cols_train - 1:].ravel()

        X_test = np.zeros((rows_test,fn),dtype=float)
        y_test = data_test[:, cols_test - 1:].ravel()

        #for i in range(0,fn):
        i = 0
        for j in range(0,cols_train-1):
            if fi[j] == True:
                X_train[:,i] = data_train[:,j]
                X_test[:,i] = data_test[:,j]
                i += 1


        df_train_X = pd.DataFrame(X_train)
        df_train_y = pd.DataFrame(y_train)
        df_test_X = pd.DataFrame(X_test)
        df_test_y = pd.DataFrame(y_test)
        df_train_X.to_csv('df_train_X.csv')
        df_train_y.to_csv('df_train_y.csv')
        df_test_X.to_csv('df_test_X.csv')
        df_test_y.to_csv('df_test_y.csv')

        rt,ct = X_train.shape
        re,ce = X_test.shape
        parameters = {}
        SVM = LinearSVC()
        grid_search_cv = GridSearchCV(SVM, parameters, cv=3, n_jobs=-1, return_train_score=True, refit=True,
                                      verbose=0)
        grid_search_cv.fit(X_train, y_train)
        resultsdf = pd.DataFrame(grid_search_cv.cv_results_)

        train_score = grid_search_cv.score(X_train, y_train)
        print('Final Linear SVM Train Score : ', train_score)
        pred = grid_search_cv.best_estimator_.predict(X_test)
        score = accuracy_score(pred, y_test)
        print('Final Linear SVM Accuracy: ',score)


    def runCHA(self):
        self.loadFeatures()
        self.executeFeatureSelection()
        self.Linear_SVM()



'''
if __name__ == '__main__':
    print('******************************')
    print('[%s] : Start' % datetime.now())
    print('******************************')
    cha = CHA()
    cha.runCHA()
    print('******************************')
    print('[%s] : End' % datetime.now())
    print('******************************') '''






