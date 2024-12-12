# A flax.nnx Implementation for Densely Connected Convolutional Networks (DenseNets)

This repository contains a [Flax.nnx](https://github.com/google/flax) implementation of the
paper [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993). The code is based on the
excellent [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html) by Phillip Lippe
with update to use new flax.nnx instead of flax.linen.

## DenseNets

[DenseNets [1]](https://arxiv.org/abs/1608.06993) were introduced in late 2016. DenseNets address this shortcoming by
reducing the size of the modules and by introducing more connections between layers. In fact, the output of each layer
flows directly as input to all subsequent layers of the same feature dimension as illustrated in their Figure 1 (below).
This increases the dependency between the layers and thus reduces redundancy.

The benchmark code trains a DenseNet on CIFAR 10.

### Cite

If you use DenseNets in your work, please cite the original paper as:

```
@article{Huang2016Densely,
  author  = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
  title   = {Densely Connected Convolutional Networks},
  journal = {arXiv preprint arXiv:1608.06993},
  year    = {2016}
}
```

### References

[1] Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2016). Densely connected convolutional networks. arXiv
preprint arXiv:1608.06993.
