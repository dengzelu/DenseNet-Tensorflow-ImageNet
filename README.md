# DenseNet-ImageNet pure tensorflow !

### Introduction
We convert the [caffemodel](https://github.com/shicai/DenseNet-Caffe) to .npy weights file (now only DenseNet_161 is available)

For more details about this network, visit [this](https://arxiv.org/abs/1608.06993)

### something about the labels
Note that there two label strategy for ImageNet data. One is the official label, see DenseNet_161/resources/official_order.txt and the other is WNID ascending order, see DenseNet_161/resources/ascending_order.txt. Actually, the latter is much more popular than the former. Most of the weights pretrained on ImageNet use the WNID ascending order label strategy. I mention this problem just to show the differnece between two label strategy. (also because I use the the official label and I am confused at first. I load weights and predict a label for an image but all wrong)

### download
DenseNet_161 weights file [BaiduDisk](http://pan.baidu.com/s/1i4DLPGD)
