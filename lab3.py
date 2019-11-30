import math
import random

size = 100

e = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

o = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
     0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
     0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
     1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
     1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
     1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
     1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
     0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
     0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
     0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

sh = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
      1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
      1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def fix(vect):
    for i in range(size):
        if vect[i] == 0:
            vect[i] = -1


def printLetter(array):
    for i in range(int(math.sqrt(size))):
        for j in range(int(math.sqrt(size))):
            print(array[i * int(math.sqrt(size)) + j], end='')
        print()
    print()


def calculateMatrix():
    result = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            result[i][j] = e[i] * e[j] + o[i] * o[j] + sh[i] * sh[j]
            if i == j:
                result[i][j] = 0
    return result


def multVectMatrix(W, vect):
    result = [0 for i in range(size)]
    for i in range(size):
        for j in range(size):
            result[i] += vect[j] * W[i][j]
    for i in range(size):
        if result[i] > 0:
            result[i] = 1
        else:
            result[i] = -1
    return result


def compare(vect):
    eState = True
    oState = True
    shState = True
    for i in range(size):
        if vect[i] != e[i]:
            eState = False
        if vect[i] != o[i]:
            oState = False
        if vect[i] != sh[i]:
            shState = False
        if eState is False and oState is False and shState is False:
            return [0, 0, 0]
    return [eState, oState, shState]


def noise(vect, persent):
    vectCopy = vect.copy()
    noised = []
    for i in range(persent):
        noisedIndex = random.randint(0, size - 1)
        if not bool(noised.count(noisedIndex)):
            noised.append(noisedIndex)
            if vectCopy[noisedIndex] == -1:
                vectCopy[noisedIndex] = 1
            else:
                vectCopy[noisedIndex] = -1
        else:
            i -= 1
    return vectCopy


def figureOut(W, vect, breakpnt=100):
    while breakpnt:
        breakpnt -= 1
        temp = multVectMatrix(W, vect)
        cmp = compare(temp)
        if bool(cmp.count(1)):
            print("Got")
            printLetter(vect)
            print("Found")
            if cmp[0] == 1:
                printLetter(e)
                return 'e'
            if cmp[1] == 1:
                printLetter(o)
                return 'o'
            if cmp[2] == 1:
                printLetter(sh)
                return 'sh'
    return 'error'


printLetter(e)
printLetter(o)
printLetter(sh)
fix(o)
fix(e)
fix(sh)
W = calculateMatrix()
noiseQuantity = 0
while noise != 100:
    temp = noise(e, noiseQuantity)
    character = figureOut(W, temp)
    print(character, end='')
    print(noiseQuantity)
    if character == 'error':
        break
    noiseQuantity += 10

noiseQuantity = 0
while noiseQuantity != 100:
    temp = noise(o, noiseQuantity)
    character = figureOut(W, temp)
    print(character, end='')
    print(noiseQuantity)
    if character == 'error':
        break
    noiseQuantity += 10

noiseQuantity = 0
while noiseQuantity <= 100:
    temp = noise(sh, noiseQuantity)
    character = figureOut(W, temp)
    print(character, end='')
    print(noiseQuantity)
    if character == 'error':
        break
    noiseQuantity += 10
print('1')
