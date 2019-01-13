import numpy as np
import sys
sys.path.append("./core/")
from dense import Dense
from sigmoid import Sigmoid
from softmax import Softmax
from data_loader import get_images_and_labels, get_test_images_and_labels
import random

if __name__ == '__main__':
    
    dense = Dense(10, 784)
    dense.load_model("./model/w.npy", "./model/b.npy")
    sigmoid = Sigmoid()
    loss = Softmax()
    
    img, labels = get_images_and_labels()
    test_imgs, test_label = get_test_images_and_labels()
    
    train_label = np.zeros([10, 1])
    train_label[labels[0]] = 1
    inputx = (img[0] - 128) / 256.0
    
    batch_size = 1;
    stop_accuracy_rate = 0.9;
    
    image_number = 60000
    for k in range(3000):
        index_list = [i for i in range(image_number)]
        random.shuffle(index_list)
        
        for i in range(image_number // batch_size):
            train_image = np.zeros((784, batch_size))
            train_label = np.zeros((10, batch_size))
            
            for j in range(batch_size):
                index = index_list[i * batch_size + j]
                inputx = (img[index] - 128) / 256.0
                inputx = inputx.reshape((784, 1))
                label = np.zeros([10, 1])
                label[labels[index]] = 1
                train_image[:, j:j + 1] = inputx
                train_label[:, j:j + 1] = label
                
                
            dense.forward(train_image)
            sigmoid.forward(dense.end)
            loss.forward(sigmoid.end)
            loss.backward(train_label)
            sigmoid.backward(loss.grad)
            dense.backward(sigmoid.grad)
    
        print("--------------")
        count = 0
        for i in range(10000):
            inputx = (test_imgs[i] - 128) / 256.0
            inputx = inputx.reshape((784, 1))
            dense.forward(inputx)
            sigmoid.forward(dense.end)
            loss.forward(sigmoid.end)
            
            if (loss.end.argmax() == test_label[i]):
                count += 1
                
        print("epoch = ", k, "   ", count / 10000)
        
        if count / 10000 > stop_accuracy_rate:
            np.save("./model/w.npy", dense.w)
            np.save("./model/b.npy", dense.b)
            break
