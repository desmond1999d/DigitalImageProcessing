import math
import random

from scipy.spatial import distance

SIDE = 3
n = 36
h = 6
m = 5
sampleLength = 5
a = 0.05
b = a

L = [1, 1, 0, 0, 1, 1,
     1, 1, 1, 0, 1, 1,
     1, 1, 1, 0, 1, 1,
     1, 1, 1, 1, 1, 1,
     1, 1, 0, 1, 1, 1,
     1, 1, 0, 1, 1, 1]

dL = [0, 0, 0, 0, 1]

U = [1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1,
     1, 1, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 0,
     1, 1, 1, 1, 1, 0,
     1, 1, 0, 0, 0, 0]

dU = [0, 0, 0, 1, 0]

Tl = [0, 1, 1, 1, 1, 0,
      0, 0, 1, 1, 0, 0,
      0, 0, 1, 1, 0, 0,
      0, 0, 1, 1, 0, 0,
      0, 0, 1, 1, 0, 0,
      0, 1, 1, 1, 1, 0]

dT = [0, 0, 1, 0, 0]

O = [1, 1, 1, 1, 1, 0,
     1, 1, 0, 0, 0, 1,
     1, 1, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 0,
     1, 1, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0]

dO = [0, 1, 0, 0, 0]

K = [1, 1, 1, 1, 0, 0,
     1, 1, 1, 1, 1, 0,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 1, 1, 1, 0,
     1, 1, 1, 1, 0, 0]

dK = [1, 0, 0, 0, 0]

inData = [[L, dL, 'N'], [U, dU, 'F'], [Tl, dT, 'I'], [O, dO, 'P'], [K, dK, 'D']]
dData = [dL, dU, dT, dO, dK]
lettersData = [L, U, Tl, O, K]


def calcD(Y, answer):
    result = [0 for i in range(m)]
    for i in range(m):
        result[i] = answer[i] - Y[i]
    return result


def calcError(Y, answer):
    E = []
    for i in range(m):
        E.append(math.fabs(Y[i] - answer[i]))
    return max(E)


def initW():
    w = []
    w1 = []
    w2 = []
    for i in range(n):
        temp = [random.uniform(-1, 1) for j in range(h)]
        w1.append(temp)
    w.append(w1)
    for i in range(h):
        temp = [random.uniform(-1, 1) for j in range(m)]
        w2.append(temp)
    w.append(w2)
    return w


def initQ():
    return [random.uniform(-0.5, 0.5) for i in range(h)]


def initT():
    return [random.uniform(-0.5, 0.5) for i in range(m)]


def calcW(W, Y, D, G):
    Wtemp = W.copy()
    for i in range(h):
        for j in range(m):
            Wtemp[i][j] = W[i][j] + a * Y[j] * (1 - Y[j]) * D[j] * G[i]
    return Wtemp


def calcV(V, G, E, X):
    Vtemp = V.copy()
    for i in range(n):
        for j in range(h):
            Vtemp[i][j] = V[i][j] + b * G[j] * (1 - G[j]) * E[j] * X[i]
    return Vtemp


def calcQ(Q, G, E):
    Qtemp = Q.copy()
    for i in range(h):
        Qtemp[i] = Q[i] + b * G[i] * (1 - G[i]) * E[i]
    return Qtemp


def calcT(T, Y, D):
    Ttemp = T.copy()
    for i in range(m):
        Ttemp[i] = T[i] + a * Y[i] * (1 - Y[i]) * D[i]
    return Ttemp


def calcE(W, Y, D):
    result = [0 for i in range(h)]
    for i in range(h):
        for j in range(m):
            result[i] += D[j] * Y[j] * (1 - Y[j]) * W[i][j]
    return result


def calcS(W, G, T, index):
    result = 0
    for i in range(h):
        result += W[i][index] * G[i]
    return result + T[index]


def calcG(V, X, Q):
    result = [0 for i in range(h)]
    for j in range(h):
        for i in range(n):
            result[j] += V[i][j] * X[i]
        result[j] += Q[j]
        result[j] = 1 / (1 + math.e ** (-1 * result[j]))
    return result


def calcY(W, G, T):
    result = [0 for i in range(m)]
    for j in range(m):
        for i in range(h):
            result[j] += W[i][j] * G[i]
        result[j] += T[j]
        result[j] = 1 / (1 + math.e ** (-1 * result[j]))
    return result


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


def find(V, W, X, Q, T):
    G = calcG(V, X, Q)
    Y = calcY(W, G, T)
    result = processResult(inData, Y)
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


def study():
    iters = 0
    W = initW()
    V = W[0]
    W = W[1]
    Q = initQ()
    T = initT()
    maxError = 0.05
    isError = True
    wasTrained = [False for i in range(sampleLength)]
    while isError:
        for i in range(sampleLength):
            X = lettersData[i]
            answer = dData[i]
            G = calcG(V, X, Q)
            Y = calcY(W, G, T)
            D = calcD(Y, answer)
            e = calcE(W, Y, D)
            W = calcW(W, Y, D, G)
            V = calcV(V, G, e, X)
            T = calcT(T, Y, D)
            Q = calcQ(Q, G, e)
            error = calcError(Y, answer)
            if error < maxError:
                wasTrained[i] = True
        isError = False
        iters += 1
        for i in range(sampleLength):
            if not wasTrained[i]:
                isError = True
        if not isError:
            print(iters)
    return V, W, Q, T


def test(V, W, Q, T):
    for i in range(sampleLength):
        for j in range(0, 100, 5):
            print(inData[i][2], end=' ')
            noised = noise(inData[i][0], j)
            found = find(V, W, noised, Q, T)
            print(j)
            if found != inData[i][2]:
                break


Vtrained, Wtrained, Qtrained, Ttrained = study()
test(Vtrained, Wtrained, Qtrained, Ttrained)
