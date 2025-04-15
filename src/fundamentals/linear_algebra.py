import numpy as np

class LinearAlgebra:
    @staticmethod
    def dot_product(a, b):
        """Compute dot product of two vectors"""
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def matrix_multiply(A, B):
        """Multiply two matrices"""
        if len(A[0]) != len(B):
            raise ValueError("Number of columns in A must equal number of rows in B")
        
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    @staticmethod
    def transpose(matrix):
        """Compute transpose of a matrix"""
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    @staticmethod
    def vector_norm(vector):
        """Compute L2 norm of a vector"""
        return np.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def matrix_inverse(matrix):
        """Compute inverse of a square matrix"""
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix must be square")
        
        # Convert to numpy array for computation
        matrix = np.array(matrix)
        return np.linalg.inv(matrix).tolist() 