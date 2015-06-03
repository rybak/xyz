#!/usr/bin/python3

from PIL import Image, ImageDraw
import scipy.spatial as ss
import sys
import random
import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import FastICA
from scipy.stats import rankdata

eps = 1/1000000000

def fromImageToArray(image):
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
    k = [[0 for x in range(2000)] for x in range(2)]
    a = 0
    for i in range(len(data)):
        t = indexes[i]
        if (data[t] == 0):
            p[0][a] = t % 100
            p[1][a] = t // 100
            a += 1
    for i in range(2000):
        ind = random.randint(0, a - 1)
        k[0][i] = p[0][ind]
        k[1][i] = p[1][ind]
    return np.array(k) - 50

def showImage(new):
    newX = new.copy()
    newX += 400
    newX = newX.astype(int)
    size = (800, 800)
    newIm = Image.new('P', size, color = 255) 
    for i in range(len(newX[0])):
        newIm.putpixel((newX[0][i],newX[1][i]), 0)
    newIm.show()    

def shuffle(a):
    indexes = [i for i in range(len(a[0]))]
    random.shuffle(indexes)
    b = a.copy()
    for i in range(len(a[0])):
        a[0][indexes[i]] = b[0][i]
        a[1][indexes[i]] = b[1][i]


def mutualInformation(A, B):
    X = rankdata(A)
    X /= len(X)
    Y = rankdata(B)
    Y /= len(Y)
    res = 0;
    forTree = np.transpose(np.array([X, Y]))
    tree = ss.cKDTree(forTree)
    cnt = 0
    for i in range(len(A)):
        pp = tree.query(forTree[i],arg_S,p=2)[0] # find nearest arg_S+1 distances
        pp = np.power(pp,(1-alpha)*2)
        res += np.sum(pp)
    res = res / arg_S
    res = res / np.power(len(A),alpha)
    res = np.log(res)
    res = res / (1 - alpha)    
    return res

# default values
arg_S = 50
alpha = 0.99
# if supplied, use non-default
print(len(sys.argv))
if (len(sys.argv) == 3):
    arg_S = int(sys.argv[1])
    alpha = float(sys.argv[2])
print("S = {:d} , alpha = {:.3f}".format(arg_S, alpha))

imX = fromImageToArray('x.bmp')
imY = fromImageToArray('y.bmp')
imZ = fromImageToArray('z.bmp')

S = np.concatenate((imX, imY, imZ), axis = 0)

mat = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        mat[i][j] = random.random()*2 - 1

S = mat.dot(S)
Y = np.vsplit(S, np.array([2, 4]))
ica = FastICA(n_components = 6, max_iter=2000, random_state = 23)

ans = ica.fit_transform(np.transpose(S))


S = np.transpose(ans)
S = S * 1000
best = 1000000000000


for i in range(1,6):
    S[i], S[1] = S[1].copy(), S[i].copy()
    for j in range(3,6):
        S[j], S[3] = S[3].copy(), S[j].copy()
        mi = mutualInformation(S[0], S[1])+mutualInformation(S[2], S[3])+mutualInformation(S[4], S[5])
        print(mi)
        if mi < best:
            best = mi
            l = S.copy()
        S[3], S[j] = S[j].copy(), S[3].copy()
    S[1], S[i] = S[i].copy(), S[1].copy()
 

Y = np.vsplit(l, np.array([2, 4]))
showImage(Y[0])
showImage(Y[1])
showImage(Y[2])
print("Best = ",best)

