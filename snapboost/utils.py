import numpy as np


class MeanSquaredError:
    """
    Mean Squared Error Loss
    """
    @staticmethod
    def compute_derivatives(y, f):
        g = 2*(f - y)
        h = 2.0 * np.ones(y.shape[0])
        return g, h


class Logistic:
    """
    Logistic Loss
    """
    @staticmethod
    def compute_derivatives(y, f):
        tmp = np.exp(-np.multiply(y, f))
        tmp2 = np.divide(tmp, 1+tmp)
        g = -np.multiply(y, tmp2)
        h = np.multiply(tmp2, 1.0-tmp2)
        return g, h
