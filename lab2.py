import math

import numpy as np
from PIL import Image
import cv2
from KmeansCustom import KmeansCustom


def show(img, map):
    for i in range(map.__len__()):
        for j in range(map[0].__len__()):
            img.putpixel((i, j), (255 - map[i][j] * 4, 255))
    return img


def showClustered(img, end):
    for i in range(map.__len__()):
        for j in range(map[0].__len__()):
            img.putpixel((i, j), (255 - map[i][j] * 4, 255))


def compactness(sqr, per):
    return (per ** 2) / sqr


def square(map, code):
    result = 0
    for i in range(map.__len__()):
        for j in range(map[0].__len__()):
            if map[i][j] == code:
                result += 1
    return result


def erode(imgTMP, iters):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(imgTMP, kernel, iterations=iters)


def dilate(imgTMP, iters):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(imgTMP, kernel, iterations=iters)


def perimeter(map, code):
    result = 0
    crop = [[0 for z in range(map[0].__len__() + 1)] for x in range(map.__len__() + 1)]
    for i in range(map.__len__()):
        for j in range(map[0].__len__()):
            crop[i + 1][j + 1] = map[i][j]
    for i in range(1, map.__len__() - 1):
        for j in range(1, map[0].__len__() - 1):
            if crop[i][j] == code and \
                    (crop[i - 1][j] != code
                     or crop[i + 1][j] != code
                     or crop[i][j - 1] != code
                     or crop[i][j + 1] != code):
                result += 1
    return result


def getDiscreteCenter(map, code, r, k):
    massX, massY = massCenter(map, code)
    result = 0
    for i in range(map.__len__()):
        for j in range(map[0].__len__()):
            if map[i][j] == code:
                result += ((j - massX) ** r) * ((i - massY) ** k)
    return result


def elongation(map, code):
    m20 = getDiscreteCenter(map, code, 2, 0)
    m02 = getDiscreteCenter(map, code, 0, 2)
    m11 = getDiscreteCenter(map, code, 1, 1)
    resultTop = (m20 + m02 + math.sqrt((m20 - m02) ** 2 + 4 * (m11 ** 2)))
    resultBotom = (m20 + m02 - math.sqrt((m20 - m02) ** 2 + 4 * (m11 ** 2)))
    result = resultTop / resultBotom
    return result


def massCenter(map, code):
    massX = 0
    massY = 0
    s = 0
    for i in range(map.__len__()):
        for j in range(map[0].__len__()):
            if map[i][j] == code:
                massX += j
                massY += i
                s += 1
    massX /= s
    massY /= s
    return massX, massY


def binarization(image):
    for i in range(0, img.size[0]):
        for j in range(0, img.size[1]):
            f = img.getpixel((i, j))
            if f[0] > 200:
                pixel = 255
            else:
                pixel = 0
            img.putpixel((i, j), (pixel, 255))
    return image


def pixelBinding(image):
    ySize = img.size[0]
    xSize = img.size[1]
    codes = []
    coloredMap = [[0 for z in range(xSize)] for x in range(ySize)]
    cur = 1
    for i in range(ySize):
        for j in range(xSize):
            kn = j - 1
            if kn < 0:
                B = (0, 0)
            else:
                B = image.getpixel((i, kn))
            km = i - 1
            if km < 0:
                C = (0, 0)
            else:
                C = image.getpixel((km, j))
            A = image.getpixel((i, j))
            if A[0] == 0:
                continue
            if B[0] == 0 and C[0] == 0:
                cur += 1
                coloredMap[i][j] = cur
                codes.append(cur)
            elif B[0] != 0 and C[0] == 0:
                coloredMap[i][j] = coloredMap[i][j - 1]
            elif B[0] == 0 and C[0] != 0:
                coloredMap[i][j] = coloredMap[i - 1][j]
            elif B[0] != 0 and C[0] != 0:
                coloredMap[i][j] = coloredMap[i - 1][j]
                if coloredMap[i - 1][j] != coloredMap[i][j - 1]:
                    codes.remove(coloredMap[i][j - 1])
                    for k in range(i + 1):
                        for l in range(xSize):
                            if coloredMap[k][l] == coloredMap[i][j - 1]:
                                coloredMap[k][l] = coloredMap[i - 1][j]
    return coloredMap, codes


# def pixelBindingDemo(image):
#     xSize = len(image[0])
#     ySize = len(image)
#     coloredMap = [[0 for z in range(xSize)] for x in range(ySize)]
#     cur = 1
#     for i in range(ySize):
#         for j in range(xSize):
#             kn = j - 1
#             if kn < 0:
#                 B = 0
#             else:
#                 B = image[i][kn]
#             km = i - 1
#             if km < 0:
#                 C = 0
#             else:
#                 C = image[km][j]
#             A = image[i][j]
#             if A == 0:
#                 continue
#             if B == 0 and C == 0:
#                 cur += 1
#                 coloredMap[i][j] = cur
#             elif B != 0 and C == 0:
#                 coloredMap[i][j] = coloredMap[i][j - 1]
#             elif B == 0 and C != 0:
#                 coloredMap[i][j] = coloredMap[i - 1][j]
#             elif B != 0 and C != 0:
#                 coloredMap[i][j] = coloredMap[i - 1][j]
#                 if coloredMap[i - 1][j] != coloredMap[i][j - 1]:
#                     for k in range(i + 1):
#                         for l in range(xSize):
#                             if coloredMap[k][l] == coloredMap[i][j - 1]:
#                                 coloredMap[k][l] = coloredMap[i - 1][j]
#     return coloredMap


imageFileName = "P0.jpg"
img = Image.open(imageFileName).convert('LA')
img.save('P1.png')
img = cv2.imread('P1.png', 0)

img = erode(img, 3)
img = dilate(img, 3)
cv2.imwrite('P3.png', img)
img = Image.open('P3.png').convert('LA')
img = binarization(img)
img.show()
bindedImage, codes = pixelBinding(img)
vectors = [[0, 0, 0, 0, 0] for i in range(codes.__len__())]
for i in range(codes.__len__()):
    sqr = square(bindedImage, codes[i])
    per = perimeter(bindedImage, codes[i])
    compactnessValue = compactness(sqr, per)
    elongationValue = elongation(bindedImage, codes[i])
    print('sqr = ')
    print(sqr)
    print('per = ')
    print(per)
    print('compactness = ')
    print(compactnessValue)
    print('elongation = ')
    print(elongationValue)
    print('')
    vectors[i][0] = sqr
    vectors[i][1] = per
    vectors[i][2] = compactnessValue
    vectors[i][3] = elongationValue
    vectors[i][4] = codes[i]

kmeans = KmeansCustom(3, vectors[0].__len__(), vectors)
end = kmeans.calculate()
img = show(img, bindedImage)
img.show()
print(end)
print('1')
