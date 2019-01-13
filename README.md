# mnist_easy_net
这也许是最简单的mnist分类网络实现方式

mnist_easy_net代码实现了对mnist数据集的分类，通过Python中numpy库实现了全连接层、sigmoid层以及softmax层，其核心代码不足80行。
最近几年深度学习的兴起导致越来越多优秀的深度学习框架的诞生，包括caffe、mxnet、pytoch、tensorflow等。以上几种框架在构建神经网络
模型时十分简单，但是这种建模方式往往无需知道神经网络的内部运行机制。

为了更加深入的理解神经网络的内部运行机制，当前代码通过mnist数据集、Python语言，其核心代码不足80行的前提下实现了对mnist手写数据集
的分类，大家可以通过该项目以一种较为简单的方式来理解神经网络的内部运行机制。

使用方法如下：

1、当前代码使用的Python版本号为3.6.4。

2、下载代码到本地，可以点击网页中的Clone or download按钮然后再点击Download ZIP实现对当前工程的下载。如果系统中安装了git工具，也可以通过命令：git clone --recursive https://github.com/suguliwei/mnist_easy_net 。

3、将data文件夹下四个文件解压到data文件夹下。

4、运行python test.py命令测试model文件夹下模型的准确率或者运行python train.py命令实现模型的训练。

5、model文件夹下存放的是已训练好的模型文件，它在mnist测试集上的分类正确率为90.07%，data文件夹下存放的是mnist数据集，core文件夹
下存放的是各个神经网络层的实现。

