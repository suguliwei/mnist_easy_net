import numpy as np

class Sigmoid():
    
    def forward(self, x):
        self.end = 1.0 / (1 + np.exp(-1 * x))
        
    def backward(self, grad):
        self.grad = grad * self.end * (1.0 - self.end)
        