import random
import math
import numpy as np

class Dense():
    
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.lr = 0.1
        self.xavier_init()
    
    def forward(self, x):
        self.x = x
        self.end = np.dot(self.w, x) + self.b
    
    def backward(self, grad):
        self.wdiff = np.dot(grad, self.x.T)
        self.bdiff = grad
        
        self.w -= self.lr * self.wdiff
        self.b -= self.lr * self.bdiff
        
        self.grad = np.dot(self.w.T, grad)
        
    def load_model(self, w, b):
        self.w = np.load(w)
        self.b = np.load(b)
    
    def xavier_init(self):
        left = math.sqrt(6.0 / (self.row + self.col))
        self.w = np.zeros((self.row, self.col))
        for i in range(self.row):
            for j in range(self.col):
                self.w[i][j] = random.uniform(-1 * left, left)
                
        self.b = np.zeros((self.row, 1))