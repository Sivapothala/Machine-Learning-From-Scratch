import numpy as np

class Metrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Calculate mean squared error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy score"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Calculate confusion matrix"""
        n_classes = len(np.unique(y_true))
        matrix = np.zeros((n_classes, n_classes))
        for i in range(len(y_true)):
            matrix[y_true[i], y_pred[i]] += 1
        return matrix
    
    @staticmethod
    def precision(y_true, y_pred):
        """Calculate precision score"""
        cm = Metrics.confusion_matrix(y_true, y_pred)
        return np.diag(cm) / np.sum(cm, axis=0)
    
    @staticmethod
    def recall(y_true, y_pred):
        """Calculate recall score"""
        cm = Metrics.confusion_matrix(y_true, y_pred)
        return np.diag(cm) / np.sum(cm, axis=1) 