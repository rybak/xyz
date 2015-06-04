#!/usr/bin/python3

from PIL import Image, ImageDraw
import scipy.spatial as ss
import sys
import random
import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import FastICA
from scipy.stats import rankdata

# default values
arg_S = 4
alpha = 0.99
sampleSize = 2000;
# if supplied, use non-default
if (len(sys.argv) == 4):
    arg_S = int(sys.argv[1])
    alpha = float(sys.argv[2])
    sampleSize = int(sys.argv[3])
print("S = {:d}, alpha = {:.3f}, n = {:d}".format(arg_S, alpha, sampleSize))



seed = 23;
random.seed(seed)
np.random.seed(seed)
eps = 1/1000000000

def fromImageToArray(image):
    global sampleSize
    f =  Image.open(image)
    data = list(f.getdata())
    a = 0
    for i in range(len(data)):
        if (data[i] == 0):
            a += 1
    n = 100
    indexes = [i for i in range(len(data))]
    random.shuffle(indexes)
    p = [[0 for x in range(a)] for x in range(2)]
    k = [[0 for x in range(sampleSize)] for x in range(2)]
    a = 0
    for i in range(len(data)):
        t = indexes[i]
        if (data[t] == 0):
            p[0][a] = t % 100
            p[1][a] = t // 100
            a += 1
    putImage(np.array(p), isNorm = False)
    for i in range(sampleSize):
        ind = random.randint(0, a - 1)
        k[0][i] = p[0][ind]
        k[1][i] = p[1][ind]
    return np.array(k) - 50


picLen = 120
bordLen = 20
rowLen = 3;
newIm = Image.new('P', (rowLen*picLen, rowLen*picLen), color = 255)
imagesAdded = 0

def normalize(a):
    a = a - a.min()
    a = a / a.max()
    a = a * (picLen - bordLen)
    return a + bordLen // 2


def putImage(new, isNorm = True):
    global imagesAdded
    newX = new.copy()
    if isNorm:
        newX[0] = normalize(newX[0])
        newX[1] = normalize(newX[1])
    newX = newX.astype(int)
    newX[0] += imagesAdded % rowLen * picLen
    newX[1] += imagesAdded // rowLen * picLen
    imagesAdded += 1
    for i in range(len(newX[0])):
        newIm.putpixel((newX[0][i], newX[1][i]), 0) # 0 == black color

def showImage():
    newIm.show()
    newIm.save("S={:d}_alpha={:.3f}_n={:d}.bmp".format(arg_S, alpha, sampleSize), 'bmp')


def evalNearest(forTree):
    tree = ss.cKDTree(forTree)
    res = 0
    for i in range(len(forTree)):
        pp = tree.query(forTree[i], arg_S + 1, p=2)[0] # find nearest arg_S distances
        pp = np.power(pp, (1-alpha)*2)
        res += np.sum(pp)
    return res


Gamma = evalNearest(np.random.rand(sampleSize*10, 2)) / np.power(sampleSize*10, alpha)
print("Gamma = ", Gamma)


def mutualInformation(A, B):
    X = rankdata(A) / len(A)
    Y = rankdata(B) / len(B)
    forTree = np.transpose(np.array([X, Y]))
    res = evalNearest(forTree)
    res = res / Gamma / np.power(len(A), alpha)
    res = np.log(res)
    res = res / (1 - alpha)
    return res



imX = fromImageToArray('x.bmp')
imY = fromImageToArray('y.bmp')
imZ = fromImageToArray('z.bmp')


S = np.concatenate((imX, imY, imZ), axis = 0)

mat = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        mat[i][j] = random.random()*2 - 1

S = mat.dot(S)
Y = np.vsplit(S, np.array([2, 4]))
putImage(Y[0])
putImage(Y[1])
putImage(Y[2])
ica = FastICA(n_components = 6, max_iter=2000, random_state = seed)
ans = ica.fit_transform(np.transpose(S))


S = np.transpose(ans)


best = 1000000000000
for i in range(1,6):
    S[i], S[1] = S[1].copy(), S[i].copy()
    for j in range(3, 6):
        S[j], S[3] = S[3].copy(), S[j].copy()
        mi = mutualInformation(S[0], S[1]) + mutualInformation(S[2], S[3]) + mutualInformation(S[4], S[5])
        print(mi)
        if mi < best:
            best = mi
            l = S.copy()
        S[3], S[j] = S[j].copy(), S[3].copy()
    S[1], S[i] = S[i].copy(), S[1].copy()


Y = np.vsplit(l, np.array([2, 4]))
putImage(Y[0])
putImage(Y[1])
putImage(Y[2])
print("Best = ", best)
print("Maximum sum of mutual infromation = ", -best)
showImage()
