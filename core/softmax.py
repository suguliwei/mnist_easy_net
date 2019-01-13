import numpy as np

class Softmax():
    
    def forward(self, x):
        x_max = np.max(x)
        x = x - x_max
        exp_x = np.exp(x)
        self.end = exp_x / np.sum(exp_x)
        
        
    def backward(self, label):
        self.grad = self.end - label