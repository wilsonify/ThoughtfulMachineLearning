import numpy as np


def get_x_with_an_intercept(x_nda):
    ones = np.ones(len(x_nda)).reshape(-1, 1)
    X_with_intercept = np.hstack((ones, x_nda))
    return X_with_intercept

def get_x_sans_intercept(x_nda):
    zeros = np.zeros(len(x_nda)).reshape(-1, 1)
    X_with_intercept = np.hstack((zeros, x_nda))
    return X_with_intercept