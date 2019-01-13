import numpy as np
import sys
sys.path.append("./core/")
from dense import Dense
from sigmoid import Sigmoid
from softmax import Softmax
from data_loader import get_images_and_labels, get_test_images_and_labels

if __name__ == '__main__':
    
    dense = Dense(10, 784)
    dense.load_model("./model/w.npy", "./model/b.npy")
    dense1 = Dense(10, 100)
    sigmoid = Sigmoid()
    loss = Softmax()
    
    img, labels = get_images_and_labels()
    test_imgs, test_label = get_test_images_and_labels()
    
    train_label = np.zeros([10, 1])
    train_label[labels[0]] = 1
    inputx = (img[0] - 128) / 256.0
    
    count = 0
    for i in range(10000):
        inputx = (test_imgs[i] - 128) / 256.0
        inputx = inputx.reshape((784, 1))
        dense.forward(inputx)
        sigmoid.forward(dense.end)
        loss.forward(sigmoid.end)
        
        if (loss.end.argmax() == test_label[i]):
            count += 1
            
    print("epoch = ", "   ", count / 10000)
    
