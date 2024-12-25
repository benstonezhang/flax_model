# A flax.nnx Implementation for Variance Models

This repository contains a [Flax.nnx](https://github.com/google/flax) implementation of variance models. The code is based on the
excellent [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html) by Phillip Lippe
with update to use new flax.nnx instead of flax.linen.

## DenseNets

[DenseNets [1]](https://arxiv.org/abs/1608.06993) were introduced in late 2016. DenseNets address this shortcoming by
reducing the size of the modules and by introducing more connections between layers. In fact, the output of each layer
flows directly as input to all subsequent layers of the same feature dimension as illustrated in their Figure 1 (below).
This increases the dependency between the layers and thus reduces redundancy.

The benchmark code trains a models on CIFAR 10.
