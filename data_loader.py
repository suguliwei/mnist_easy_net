import numpy as np
import struct
 
def load_image(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    
    number = head[1]
    width = head[2]
    height = head[3]

    bits = number * width * height
    bitsString = '>' + str(bits) + 'B'
    img = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    img = np.reshape(img, [number, width * height])

    return img, head

def load_label(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
 
    number = head[1]
    offset = struct.calcsize('>II')
 
    numString = '>' + str(number) + "B"
    label = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    label = np.reshape(label, [number])
 
    return label, head


def get_images_and_labels():
    file1= './data/train-images.idx3-ubyte'
    file2= './data/train-labels.idx1-ubyte'
    img, data_head = load_image(file1)
    label, labels_head = load_label(file2)
    
    return img, label



def get_test_images_and_labels():
    file1= './data/t10k-images.idx3-ubyte'
    file2= './data/t10k-labels.idx1-ubyte'
 
    img, data_head = load_image(file1)
    label, labels_head = load_label(file2)
    
    return img, label




if __name__ == "__main__":
    file1= 'train-images.idx3-ubyte'
    file2= 'train-labels.idx1-ubyte'
 
    imgs,data_head = load_image(file1)
    labels,labels_head = load_label(file2)
    
    print('data_head:',imgs.shape)
    print(labels.shape)

