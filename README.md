# A flax.nnx Implementation for Variance Models

This repository contains a [Flax.nnx](https://github.com/google/flax) implementation of variance models. The code is
based on the excellent [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html) by Phillip Lippe with update to use new flax.nnx instead of
flax.linen.

## ResNet

[ResNet](https://arxiv.org/abs/1512.03385) were introduced in late 2015, and has been the foundation for neural networks
with more than 1,000 layers. Despite its simplicity, the idea of residual connections is highly effective as it supports
stable gradient propagation through the network.

## DenseNets

[DenseNets](https://arxiv.org/abs/1608.06993) were introduced in late 2016. DenseNets address this shortcoming by
reducing the size of the modules and by introducing more connections between layers. In fact, the output of each layer
flows directly as input to all subsequent layers of the same feature dimension as illustrated in their Figure 1 (below).
This increases the dependency between the layers and thus reduces redundancy.

The benchmark code trains a models on CIFAR 10.
