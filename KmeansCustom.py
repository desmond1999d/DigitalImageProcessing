import math
import random

import numpy as np


class KmeansCustom:

    def euclideanDistance(self, point1, point2):
        if np.isnan(point2).any():
            print('FUCK')
        result = 0
        for i in range(point1.__len__()):
            result += (point1[i] - point2[i]) ** 2
        return math.sqrt(result)

    def __init__(self, clusters, dimensions, points):
        self.points = points
        self.complicity = 0
        self.clusters = clusters
        self.dimensions = dimensions
        self.centers = [[0 for i in range(dimensions)] for j in range(clusters)]
        self.clusteredPoints = [[] for i in range(clusters)]
        for i in range(clusters):
            self.centers[i] = self.points[random.randint(0, points.__len__() - 1)]

    def calculateClusterReference(self, point):
        distances = [0 for i in range(self.clusters)]
        for i in range(self.clusters):
            distances[i] += self.euclideanDistance(point, self.centers[i])
        minDistance = distances[0]
        result = 0
        for i in range(self.clusters):
            if distances[i] < minDistance:
                minDistance = distances[i]
                result = i
        return result

    def cluster(self):
        for i in range(self.clusteredPoints.__len__()):
            self.clusteredPoints[i].clear()
        for i in range(self.points.__len__()):
            self.clusteredPoints[self.calculateClusterReference(self.points[i])].append(self.points[i])

    def recalculateCenters(self):
        for i in range(self.clusteredPoints.__len__()):
            if self.clusteredPoints[i].__len__() > 0:
                self.centers[i] = np.median(np.array(self.clusteredPoints[i]), axis=0).tolist()
            if np.isnan(self.centers[i]).any():
                print('fuck')

    def calculate(self):
        self.complicity += 1
        self.cluster()
        oldCenters = self.centers.copy()
        self.recalculateCenters()
        for i in range(self.centers.__len__()):
            for j in range(self.dimensions):
                if oldCenters[i][j] != self.centers[i][j]:
                    self.calculate()
                return self.clusteredPoints
