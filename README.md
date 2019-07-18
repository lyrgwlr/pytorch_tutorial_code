# Pytorch-tutorial-code  

这是参考pytorch官网的入门教程写的代码  
cite: [deep_learning_60min_blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)  
  
## demo1.py  
主要是一个基本的CNN网络的训练流程：  
* 定义神经网络以及要学习的参数/权重  
* 在数据集上迭代  
* 将输入经过这个网络  
* 计算损失（输出离正确值有多远）  
* 反向传播各个参数的梯度  
* 更新网络权重，一般使用 weight = weight - learning_rate * gradient 或pytorch自带的优化器  
  
## demo2.py
是一个用pytorch的CNN网络形式解方程的小例子  
  
## classifier.py
训练一个分类器：
* Load and normalizing the CIFAR10 training and test datasets using *torchvision*  
* Define a Convolutional Neural Network  
* Define a loss function  
* Train the network on the training data  
* Test the network on the test data  