

import numpy as np


class Tanh():
    @staticmethod
    def func(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x) ** 2

class Sigmoid():
    @staticmethod
    def func(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

class Linear():
    @staticmethod
    def func(x):
        return x
    
    @staticmethod
    def derivative(x):
        return 1

class Relu():
    @staticmethod
    def func(x):
        return np.where(x < 0, 0, x)
    
    @staticmethod
    def derivative(x):
        return np.where(x < 0, 0, 1)

class ReluLeaky():
    # using .01
    @staticmethod
    def func(x):
        return np.where(x < 0, .01 * x, x)
    
    @staticmethod
    def derivative(x):
        return np.where(x < 0, .01, 1)