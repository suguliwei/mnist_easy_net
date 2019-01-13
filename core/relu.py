import numpy as np

class Relu():
    
    def forward(self, x):
        self.x = x
        self.end = np.where(x > 0, x, 0)
        
    def backward(self, grad):
        self.grad = np.where(self.x > 0, grad, 0)
        