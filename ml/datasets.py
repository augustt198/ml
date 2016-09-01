import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

def prepend_ones(A):
    return np.c[np.ones(A.shape[0]), A]

def crime_rates():
    fname = 'data/crime_rates.csv'
    X = np.genfromtxt(fname, delimiter=',', usecols=range(0, 5))
    X = prepend_ones(X)

    Y = np.genfromtxt(fname, delimiter=',', usecols=5)
    Y = Y.reshape((Y.shape[0], 1))

    return (X, Y)

def house_prices():
    fname = 'data/house_prices.csv'
    X = np.genfromtxt(fname, delimiter=',', usecols=0)
    m = X.shape[0]
    X = X.reshape((m, 1))
    X = np.c_[np.ones(m), X] # bias row 
    
    Y = np.genfromtxt(fname, delimiter=',', usecols=1)
    Y = Y.reshape((m, 1))

    return (X, Y)

from array import array as pyarray
import numpy as np
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10)):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    path = "data/mnist"
    if dataset == "training":
        fname_img = os.path.join(path, 'training_imgs')
        fname_lbl = os.path.join(path, 'training_labels')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'test_imgs')
        fname_lbl = os.path.join(path, 'test_labels')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
