import numpy as np

class FoodSource():
    def __init__(self,featureAInclusion, fitness=0,nrFeatures=0):
        self.featureInclusion = featureAInclusion
        self.fitness = fitness
        self.limit = 0
        self.nrFeatures = nrFeatures


    def __repr__(self):
        return "FoodSource (featureAInclusion,fitness,nrFeatures): " \
               "(%s,%s,%s)" % (self.featureInclusion,self.nrFeatures,self.fitness)

    def __eq__(self, other):
        eq = False
        if other.nrFeatures == self.nrFeatures:
            eq = True
        return eq

    def __hash__(self):
        return hash(self.__repr__())

    def __cmp__(self, other):
        res = self.fitness - other.fitness
        if res < 0:
            return -1
        elif res > 0:
            return 1
        else:
            return 0

    def __le__(self, other):
        return (self.fitness <= other.fitness)

    def __ge__(self, other):
        return (self.fitness >= other.fitness)



    def getFeatureInclusion(self):
        return self.featureInclusion

    def setFeatureInclusion(self, featureInclusion):
        self.featureInclusion = featureInclusion

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

    def getLimit(self):
        return self.limit

    def setLimit(self,limit):
        self.limit = limit

    def incrementLimit(self):
        self.limit += 1

    def getNrFeatures(self):
        return self.nrFeatures

    def setNrFeatures(self,nrFeatures):
        self.nrFeatures = nrFeatures

    def increaseNrFeatures(self):
        self.nrFeatures += 1

    