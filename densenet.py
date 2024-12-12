import functools
import typing

import jax.numpy as jnp
from flax import nnx


class DenseNet(nnx.Module):
    def __init__(self,
                 dims: typing.Literal[1, 2, 3],
                 in_features: int,
                 num_classes: int,
                 num_blocks: typing.Tuple[int, ...] | typing.List[int],
                 *,
                 init_conv_kernel_size: int = 3,
                 init_conv_strides: int = 1,
                 init_pool_kernel_size: int | None = None,
                 init_pool_strides: int | None = None,
                 bn_size: int = 2,  # Bottleneck size (factor of growth rate) for the output of the 1 convolution
                 growth_rate: int = 16,  # Number of output channels of the 3 convolution
                 act_fn: callable = nnx.relu,
                 kernel_init: nnx.Initializer = nnx.initializers.kaiming_uniform(),
                 rngs: nnx.rnglib.Rngs):
        init_conv_kernel_size = (init_conv_kernel_size,) * dims
        init_conv_strides = (init_conv_strides,) * dims
        if init_pool_kernel_size is not None and init_pool_strides is not None:
            init_pool_kernel_size = (init_pool_kernel_size,) * dims
            init_pool_strides = (init_pool_strides,) * dims

        win_size_1 = (1,) * dims
        win_size_2 = (2,) * dims
        win_size_3 = (3,) * dims

        mean_axis = tuple([-2 - i for i in range(dims)])

        class DenseLayer(nnx.Module):
            def __init__(self, c_in: int):
                self.bn1 = nnx.BatchNorm(num_features=c_in,
                                         use_running_average=True,
                                         rngs=rngs)
                self.conv1 = nnx.Conv(in_features=c_in,
                                      out_features=bn_size * growth_rate,
                                      kernel_size=win_size_1,
                                      kernel_init=kernel_init,
                                      use_bias=False,
                                      rngs=rngs)
                self.bn2 = nnx.BatchNorm(num_features=bn_size * growth_rate,
                                         use_running_average=True,
                                         rngs=rngs)
                self.conv2 = nnx.Conv(in_features=bn_size * growth_rate,
                                      out_features=growth_rate,
                                      kernel_size=win_size_3,
                                      kernel_init=kernel_init,
                                      use_bias=False,
                                      rngs=rngs)

            def __call__(self, x):
                z = self.bn1(x)
                z = act_fn(z)
                z = self.conv1(z)
                z = self.bn2(z)
                z = act_fn(z)
                z = self.conv2(z)
                x_out = jnp.concatenate([x, z], axis=-1)
                return x_out

        class DenseBlock(nnx.Module):
            def __init__(self,
                         c_in: int,
                         num_layers: int,  # Number of dense layers to apply in the block
                         ):
                self.layers = [DenseLayer(c_in=c_in + i * growth_rate) for i in range(num_layers)]

            def __call__(self, x):
                for l in self.layers:
                    x = l(x)
                return x

        class TransitionLayer(nnx.Module):
            def __init__(self,
                         c_in: int,  # In feature size
                         c_out: int,  # Out feature size
                         ):
                self.bn = nnx.BatchNorm(num_features=c_in,
                                        use_running_average=True,
                                        rngs=rngs)
                self.conv = nnx.Conv(in_features=c_in,
                                     out_features=c_out,
                                     kernel_size=win_size_1,
                                     kernel_init=kernel_init,
                                     use_bias=False,
                                     rngs=rngs)
                self.avg_pool = functools.partial(nnx.avg_pool, window_shape=win_size_2, strides=win_size_2)

            def __call__(self, x, train=True):
                x = self.bn(x)
                x = act_fn(x)
                x = self.conv(x)
                x = self.avg_pool(x)
                return x

        self.act_fn = act_fn
        c_hidden = growth_rate * bn_size  # The start number of hidden channels
        self.conv = nnx.Conv(in_features=in_features,
                             out_features=c_hidden,
                             kernel_size=init_conv_kernel_size,
                             strides=init_conv_strides,
                             kernel_init=kernel_init,
                             rngs=rngs)
        if init_pool_kernel_size is not None and init_pool_strides is not None:
            self.max_pool = functools.partial(nnx.max_pool, window_shape=win_size_3, strides=win_size_2, padding='SAME')
        else:
            self.max_pool = lambda x: x

        self.blocks = []
        for block_idx, num_layers in enumerate(num_blocks):
            self.blocks.append(
                DenseBlock(c_in=c_hidden, num_layers=num_layers)
            )
            c_hidden += num_layers * growth_rate
            if block_idx < len(num_blocks) - 1:  # Don't apply transition layer on last block
                self.blocks.append(
                    TransitionLayer(c_in=c_hidden, c_out=c_hidden // 2)
                )
                c_hidden //= 2

        self.bn = nnx.BatchNorm(num_features=c_hidden,
                                use_running_average=True,
                                rngs=rngs)
        self.linear = nnx.Linear(in_features=c_hidden,
                                 out_features=num_classes,
                                 kernel_init=kernel_init,
                                 rngs=rngs)
        self.mean_axis = mean_axis

    def __call__(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        for b in self.blocks:
            x = b(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = x.mean(axis=self.mean_axis)
        x = self.linear(x)
        return x
