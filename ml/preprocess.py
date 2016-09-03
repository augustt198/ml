import numpy as np

# data preprocessing

def rescale(arr):
    r = float(np.max(arr) - np.min(arr))
    return (arr - np.mean(arr)) / r
