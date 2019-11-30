import random

import numpy as np

n = 36
m = 5
sampleLength = 5
a = 0.05
b = a

L = [1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1]

U = [1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 1, 1, 1, 1,
     0, 1, 1, 1, 1, 0]

Tl = [1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1,
      1, 0, 1, 1, 0, 1,
      0, 0, 1, 1, 0, 0,
      0, 0, 1, 1, 0, 0,
      0, 0, 1, 1, 0, 0]

O = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1,
     0, 1, 0, 0, 1, 0,
     0, 0, 1, 1, 0, 0]

K = [1, 1, 0, 0, 1, 0,
     1, 1, 0, 1, 0, 0,
     1, 1, 1, 0, 0, 0,
     1, 1, 1, 0, 0, 0,
     1, 1, 0, 1, 1, 0,
     1, 1, 0, 0, 1, 1]

inData = [[L, 'L'], [U, 'U'], [Tl, 'T'], [O, 'O'], [K, 'K']]
lettersData = [L, U, Tl, O, K]


def initW():
    W = []
    for i in range(n):
        temp = [random.uniform(0, 1) for j in range(m)]
        W.append(temp)
    return W


def recalcW(W, X, index):
    result = W.copy()
    for i in range(n):
        result[i][index] = (W[i][index] + b * (X[i] - W[i][index]))
    return result


def calcBest(X, W, F):
    Xvect = np.array(X)
    neuronWeight = [[] for i in range(m)]
    for i in range(m):
        for j in range(n):
            neuronWeight[i].append(W[j][i])
    d = [0 for i in range(m)]
    for i in range(m):
        Wvect = np.array(neuronWeight[i])
        d[i] = np.linalg.norm(Xvect - Wvect) * F[i]
    return d.index(min(d))


def calcError(X, winnerWeightData):
    xVect = np.array(X)
    winnerWeightDataVect = np.array(winnerWeightData)
    return np.linalg.norm(xVect - winnerWeightDataVect)


def study():
    maxError = 0.01
    W = initW()
    F = [0 for i in range(m)]
    learnedLetters = [False for i in range(sampleLength)]
    while True:
        for i in range(sampleLength):
            while True:
                X = lettersData[i]
                best = calcBest(X, W, F)
                W = recalcW(W, X, best)
                F[best] += 1
                weightVect = []
                for j in range(n):
                    weightVect.append(W[j][best])
                error = calcError(X, weightVect)
                print(error)
                if error < maxError:
                    learnedLetters[i] = True
                    break
            if all(item is True for item in learnedLetters):
                return W


def noise(vect, persent):
    vectCopy = vect.copy()
    noised = []
    for i in range(persent):
        noisedIndex = random.randint(0, n - 1)
        if not bool(noised.count(noisedIndex)):
            noised.append(noisedIndex)
            if vectCopy[noisedIndex] == 0:
                vectCopy[noisedIndex] = 1
            else:
                vectCopy[noisedIndex] = 0
        else:
            i -= 1
    return vectCopy


def getResult(W, X):
    result = [0 for i in range(m)]
    for i in range(n):
        for j in range(m):
            result[j] += X[i] * W[i][j]
    return result


def test(W):
    for i in range(sampleLength):
        noisePower = 0
        cluster = 0
        while noisePower <= 100:
            sample = lettersData[i]
            noisedSample = noise(sample, noisePower)
            result = getResult(W, noisedSample)
            if noisePower == 0:
                cluster = result.index(max(result))
            elif cluster != result.index(max(result)):
                break
            print(inData[i][1], end=' ')
            print(noisePower, end=' ')
            print(result.index(max(result)))
            noisePower += 10


W = study()
test(W)
