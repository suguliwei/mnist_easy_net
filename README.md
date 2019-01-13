# mnist_easy_net
也许是最简单的mnist分类网络实现方式

mnist_easy_net代码实现了对mnist数据集的分类，通过Python中numpy库实现了全连接层、sigmoid层以及softmax层，其核心代码不足80行。
最近几年深度学习的兴起导致越来越多优秀的深度学习框架的诞生，包括caffe、mxnet、pytoch、tensorflow等。以上几种框架在构建神经网络
模型时十分简单，但是这种建模方式往往无需知道神经网络的内部运行机制。
为了更加深入的理解神经网络的内部运行机制，当前代码通过mnist数据集、Python语言，其核心代码不足80行的前提下实现了对mnist手写数据集
的分类，大家可以通过该项目以一种较为简单的方式来理解神经网络的内部运行机制。

使用方法如下：
1、当前代码使用的Python版本号为3.6.4。

2、下载代码到本地。

3、将data文件夹下的四个压缩文件解压到data文件夹下。

4、最外面的train.py文件是训练代码部分，test.py是测试代码部分。

5、model文件夹下存放的是已训练好的模型文件，它在mnist测试集上的分类正确率为90.07%。


TODO:
1、增加更多的operator，比如convolution、depthwise convolution、batch normalization、pooling等；
2、增加更多的solver，包括sgd、momentum、adam等；
