import imageio as imageio
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(histogram))
    plt.bar(index, histogram)
    plt.show()


def logTransform(c, f):
    g = c * math.log(float(1 + f), 10)
    return g


def normalize(value):
    if value < 0:
        return 0
    elif value > 255:
        return 255
    return value


# Apply logarithmic transformation for an image
def logTransformImage(image, outputMax=255, inputMax=255):
    c = outputMax / math.log(inputMax + 1, 10)
    # Read pixels and apply logarithmic transformation
    for i in range(0, img.size[0] - 1):
        for j in range(0, img.size[1] - 1):
            # Get pixel value at (x,y) position of the image
            f = img.getpixel((i, j))
            # Do log transformation of the pixel
            redPixel = round(logTransform(c, f[0]))
            greenPixel = round(logTransform(c, f[1]))
            bluePixel = round(logTransform(c, f[2]))
            # Modify the image with the transformed pixel values
            img.putpixel((i, j), (redPixel, greenPixel, bluePixel))
    return image


def robertsTransformImage(image):
    result = Image.open(imageFileName)
    xSize = image.size[0] - 1
    ySize = image.size[1] - 1
    for i in range(0, image.size[0] - 1):
        for j in range(0, image.size[1] - 1):
            topRight = [0] * 3
            bottomRight = [0] * 3
            bottomLeft = [0] * 3
            topLeft = image.getpixel((i, j))
            if j != xSize - 1 and i != 0:
                topRight = image.getpixel((i - 1, j + 1))
            if j != 0 and i != ySize - 1:
                bottomLeft = image.getpixel((i + 1, j - 1))
            if j != xSize - 1 and i != ySize - 1:
                bottomRight = image.getpixel((i + 1, j + 1))
            firstCoreResultR = normalize(topRight[0] - bottomLeft[0])
            firstCoreResultG = normalize(topRight[1] - bottomLeft[1])
            firstCoreResultB = normalize(topRight[2] - bottomLeft[2])
            secondCoreResultR = normalize(topLeft[0] - bottomRight[0])
            secondCoreResultG = normalize(topLeft[1] - bottomRight[1])
            secondCoreResultB = normalize(topLeft[2] - bottomRight[2])
            tmpR = np.int_(math.sqrt(firstCoreResultR ** 2 + secondCoreResultR ** 2))
            tmpG = np.int_(math.sqrt(firstCoreResultG ** 2 + secondCoreResultG ** 2))
            tmpB = np.int_(math.sqrt(firstCoreResultB ** 2 + secondCoreResultB ** 2))
            result.putpixel((i, j), (tmpR, tmpG, tmpB))
    return result


# Display the original image
imageFileName = "dark.jpg"
img = Image.open(imageFileName)
# img.show()

robertsTransformedImage = robertsTransformImage(img)
robertsTransformedImage.save('rob.jpg')
# robertsTransformedImage.show()
img = Image.open(imageFileName)
logTransformedImage = logTransformImage(img)
logTransformedImage.save('log.jpg')
# logTransformedImage.show()

pic = imageio.imread('photo.jpg')

gray = lambda rgb: np.dot(rgb[..., :3], [0.21, 0.72, 0.07])
grayImage = gray(pic)

grayImage = np.int_(grayImage)

histogram = [0] * 256
indexes = list(range(0, 256))

xSize = len(grayImage[0])
ySize = len(grayImage)

for i in range(ySize):
    for j in range(xSize):
        histogram[grayImage[i][j]] += 1

plot_bar_x()

pic = imageio.imread('log.jpg')

gray = lambda rgb: np.dot(rgb[..., :3], [0.21, 0.72, 0.07])
grayImage = gray(pic)

grayImage = np.int_(grayImage)

histogram = [0] * 256
indexes = list(range(0, 256))

xSize = len(grayImage[0])
ySize = len(grayImage)

for i in range(ySize):
    for j in range(xSize):
        histogram[grayImage[i][j]] += 1

plot_bar_x()

pic = imageio.imread('rob.jpg')

gray = lambda rgb: np.dot(rgb[..., :3], [0.21, 0.72, 0.07])
grayImage = gray(pic)

grayImage = np.int_(grayImage)

histogram = [0] * 256
indexes = list(range(0, 256))

xSize = len(grayImage[0])
ySize = len(grayImage)

for i in range(ySize):
    for j in range(xSize):
        histogram[grayImage[i][j]] += 1

plot_bar_x()
