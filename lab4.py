import math
import random

from scipy.spatial import distance

SIZE = 36
SIDE = 3
n = SIZE
h = 16
m = 5
sampleLength = 5
a = 0.3
b = a
maxE = 1000000

smallTest = [[2, 1, 3]]

L = [1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1]

dL = [0, 0, 0, 0, 1]

U = [1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 1, 1, 1, 1,
     0, 1, 1, 1, 1, 0]

dU = [0, 0, 0, 1, 0]

T = [1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1,
     1, 0, 1, 1, 0, 1,
     0, 0, 1, 1, 0, 0,
     0, 0, 1, 1, 0, 0,
     0, 0, 1, 1, 0, 0]

dT = [0, 0, 1, 0, 0]

O = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1,
     0, 1, 0, 0, 1, 0,
     0, 0, 1, 1, 0, 0]

dO = [0, 1, 0, 0, 0]

K = [1, 1, 0, 0, 1, 0,
     1, 1, 0, 1, 0, 0,
     1, 1, 1, 0, 0, 0,
     1, 1, 1, 0, 0, 0,
     1, 1, 0, 1, 1, 0,
     1, 1, 0, 0, 1, 1]

dK = [1, 0, 0, 0, 0]

inData = [[L, dL, 'L'], [U, dU, 'U'], [T, dT, 'T'], [O, dO, 'O'], [K, dK, 'K']]
dData = [dL, dU, dT, dO, dK]
lettersData = [L, U, T, O, K]


def noise(vect, persent):
    vectCopy = vect.copy()
    noised = []
    for i in range(persent):
        noisedIndex = random.randint(0, SIZE - 1)
        if not bool(noised.count(noisedIndex)):
            noised.append(noisedIndex)
            if vectCopy[noisedIndex] == -1:
                vectCopy[noisedIndex] = 1
            else:
                vectCopy[noisedIndex] = -1
        else:
            i -= 1
    return vectCopy


def initW():
    w = []
    w1 = []
    w2 = []
    for i in range(n):
        temp = [random.uniform(-0.5, 0.5) for j in range(h)]
        w1.append(temp)
    w.append(w1)
    for i in range(h):
        temp = [random.uniform(-0.5, 0.5) for j in range(m)]
        w2.append(temp)
    w.append(w2)
    return w


def calcS(W, X, leftLen, rightLen):
    s = []
    temp = 0
    for i in range(rightLen):
        for j in range(leftLen):
            temp += W[j][i] * X[j]
        s.append(temp)
        temp = 0
    return s


def calcY(S, a):
    y = []
    for i in range(S.__len__()):
        y.append(1 / (1 + math.e ** (-1 * a * S[i])))
    return y


def calcE(y, d, maxE):
    E = 0
    for i in range(m):
        E += math.fabs(y[i] - d[i])
    if E < maxE:
        return E
    else:
        return maxE
    # for i in range(m):
    #     E += (y[i] - d[i]) ** 2
    # return E / 2


def calcExitErrors(Y, d):
    exitErrors = [(Y[i] - d[i]) * (Y[i] * (1 - Y[i])) for i in range(m)]
    return exitErrors


def calcErrors(errorsRight, W, Y):
    errors = []
    for i in range(h):
        temp = 0
        for j in range(m):
            temp += errorsRight[j] * W[i][j]
        temp *= (Y[i] * (1 - Y[i]))
        errors.append(temp)
    return errors


def recalcW(W, errors, X, Y):
    for i in range(n):
        for j in range(h):
            W[0][i][j] += -0.65 * errors[0][j] * X[i]
    for i in range(h):
        for j in range(m):
            W[1][i][j] += -0.65 * errors[0][j] * Y[i]


def find(W, vect):
    S1 = calcS(W[0], vect, n, h)
    Y1 = calcY(S1, a)
    S2 = calcS(W[1], Y1, h, m)
    Y2 = calcY(S2, b)
    result = processResult(inData, Y2)
    print(result, end=' ')
    return result


def processResult(inData, result):
    dif = distance.euclidean(inData[0][1], result)
    num = 0
    for i in range(inData.__len__()):
        temp = distance.euclidean(inData[i][1], result)
        if temp < dif:
            dif = temp
            num = i
    return inData[num][2]


W = initW()
# W = [[[0.5, 0.3], [-0.1, -0.23], [0.2, 0.43]], [[1], [1]]]
tolerableError = 0.085
eVect = []
temp = 0
E = 0
flag = 0
while maxE >= tolerableError:
    for i in range(sampleLength):
        S1 = calcS(W[0], lettersData[i], n, h)
        Y1 = calcY(S1, a)
        S2 = calcS(W[1], Y1, h, m)
        Y2 = calcY(S2, b)
        temp = calcE(Y2, dData[i], maxE)
        errors = []
        exitErrors = calcExitErrors(Y2, dData[i])
        middleErrors = calcErrors(exitErrors, W[0], Y1)
        errors.append(middleErrors)
        errors.append(exitErrors)
        recalcW(W, errors, lettersData[i], Y1)
    if tolerableError > temp:
        print(temp)
        break
    maxE = temp
    flag += 1
    print(maxE)
    eVect = []
    E = 0

for i in range(sampleLength):
    for j in range(0, 100, 5):
        print(inData[i][2], end=' ')
        noised = noise(inData[i][0], j)
        found = find(W, noised)
        print(j)
        if found != inData[i][2]:
            break
