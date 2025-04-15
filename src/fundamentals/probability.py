import numpy as np

class Probability:
    @staticmethod
    def mean(data):
        """Calculate the mean of a dataset"""
        return sum(data) / len(data)
    
    @staticmethod
    def variance(data):
        """Calculate the variance of a dataset"""
        mean = Probability.mean(data)
        return sum((x - mean) ** 2 for x in data) / len(data)
    
    @staticmethod
    def standard_deviation(data):
        """Calculate the standard deviation of a dataset"""
        return np.sqrt(Probability.variance(data))
    
    @staticmethod
    def covariance(x, y):
        """Calculate covariance between two variables"""
        if len(x) != len(y):
            raise ValueError("Datasets must have the same length")
        mean_x = Probability.mean(x)
        mean_y = Probability.mean(y)
        return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
    
    @staticmethod
    def correlation(x, y):
        """Calculate correlation coefficient between two variables"""
        cov = Probability.covariance(x, y)
        std_x = Probability.standard_deviation(x)
        std_y = Probability.standard_deviation(y)
        return cov / (std_x * std_y) 