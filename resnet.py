import typing

from flax import nnx

# Conv initialized with kaiming int, but uses fan-out instead of fan-in mode
# Fan-out focuses on the gradient distribution, and is commonly used in ResNets
resnet_kernel_init = nnx.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

RESNET_18_BLOCKS = (2, 2, 2, 2)
RESNET_34_BLOCKS = (3, 4, 6, 3)
RESNET_50_BLOCKS = (3, 4, 6, 3)
RESNET_101_BLOCKS = (3, 4, 23, 3)
RESNET_152_BLOCKS = (3, 8, 36, 3)


def _dummy_fn(x):
    return x


class ResNet(nnx.Module):
    def __init__(self,
                 dims: typing.Literal[1, 2, 3],
                 in_features: int,
                 num_classes: int,
                 num_blocks: typing.Sequence[int],
                 c_hiddens: typing.Sequence[int],
                 pre_activate: bool = False,
                 act_fn: callable = nnx.relu,
                 kernel_init: nnx.Initializer = resnet_kernel_init,
                 *,
                 rngs: nnx.rnglib.Rngs):

        assert len(num_blocks) == len(c_hiddens)

        win_size_1 = (1,) * dims
        win_size_2 = (2,) * dims
        win_size_3 = (3,) * dims

        mean_axis = tuple([-2 - i for i in range(dims)])

        class ResNetBlock(nnx.Module):
            def __init__(self, c_in: int, c_out: int, sub_sample: bool = False):
                self.conv1 = nnx.Conv(in_features=c_in,
                                      out_features=c_out,
                                      kernel_size=win_size_3,
                                      strides=win_size_1 if sub_sample is False else win_size_2,
                                      kernel_init=kernel_init,
                                      use_bias=False,
                                      rngs=rngs)
                self.bn1 = nnx.BatchNorm(num_features=c_out,
                                         rngs=rngs)
                self.conv2 = nnx.Conv(in_features=c_out,
                                      out_features=c_out,
                                      kernel_size=win_size_3,
                                      kernel_init=kernel_init,
                                      use_bias=False,
                                      rngs=rngs)
                self.bn2 = nnx.BatchNorm(num_features=c_out,
                                         rngs=rngs)
                self.sub_sample = nnx.Conv(in_features=c_in,
                                           out_features=c_out,
                                           kernel_size=win_size_1,
                                           strides=win_size_2,
                                           kernel_init=kernel_init,
                                           rngs=rngs) if sub_sample is True else _dummy_fn

            def __call__(self, x):
                z = self.conv1(x)
                z = self.bn1(z)
                z = act_fn(z)
                z = self.conv2(z)
                z = self.bn2(z)
                r = self.sub_sample(x)
                out = act_fn(z + r)
                return out

        class PreActResNetBlock(nnx.Module):
            def __init__(self, c_in: int, c_out: int, sub_sample: bool = False):
                self.bn1 = nnx.BatchNorm(num_features=c_in,
                                         rngs=rngs)
                self.conv1 = nnx.Conv(in_features=c_in,
                                      out_features=c_out,
                                      kernel_size=win_size_3,
                                      strides=win_size_1 if sub_sample is False else win_size_2,
                                      kernel_init=kernel_init,
                                      use_bias=False,
                                      rngs=rngs)
                self.bn2 = nnx.BatchNorm(num_features=c_out,
                                         rngs=rngs)
                self.conv2 = nnx.Conv(in_features=c_out,
                                      out_features=c_out,
                                      kernel_size=win_size_3,
                                      kernel_init=kernel_init,
                                      use_bias=False,
                                      rngs=rngs)
                if sub_sample:
                    self.bn3 = nnx.BatchNorm(num_features=c_in,
                                             rngs=rngs)
                    self.conv3 = nnx.Conv(in_features=c_in,
                                          out_features=c_out,
                                          kernel_size=win_size_1,
                                          strides=win_size_2,
                                          kernel_init=kernel_init,
                                          use_bias=False,
                                          rngs=rngs)

                    def sub_sample_fn(x):
                        x = self.bn3(x)
                        x = act_fn(x)
                        x = self.conv3(x)
                        return x

                    self.sub_sample = sub_sample_fn
                else:
                    self.sub_sample = _dummy_fn

            def __call__(self, x):
                z = self.bn1(x)
                z = act_fn(z)
                z = self.conv1(z)
                z = self.bn2(z)
                z = act_fn(z)
                z = self.conv2(z)
                r = self.sub_sample(x)
                x_out = z + r
                return x_out

        self.conv = nnx.Conv(in_features=in_features,
                             out_features=c_hiddens[0],
                             kernel_size=win_size_3,
                             kernel_init=kernel_init,
                             use_bias=False,
                             rngs=rngs)

        if pre_activate is False:
            block_cls = ResNetBlock
            self.bn = nnx.BatchNorm(num_features=c_hiddens[0],
                                    rngs=rngs)

            def first_conv_fn(x):
                x = self.conv(x)
                x = self.bn(x)
                x = act_fn(x)
                return x

            self.first_conv = first_conv_fn
        else:
            block_cls = PreActResNetBlock
            # If pre-activation block, we do not apply non-linearities yet
            self.first_conv = self.conv

        # Creating the ResNet blocks
        self.blocks = []
        for block_idx, block_count in enumerate(num_blocks):
            for bc in range(block_count):
                if bc == 0 and block_idx > 0:
                    sub_sample = True
                    c_in = c_hiddens[block_idx - 1]
                else:
                    sub_sample = False
                    c_in = c_hiddens[block_idx]

                # ResNet block
                block = block_cls(c_in=c_in,
                                  c_out=c_hiddens[block_idx],
                                  sub_sample=sub_sample)
                self.blocks.append(block)

        self.linear = nnx.Linear(in_features=c_hiddens[-1],
                                 out_features=num_classes,
                                 rngs=rngs)
        self.mean_axis = mean_axis

    def __call__(self, x):
        # A first convolution on the original image to scale up the channel size
        x = self.first_conv(x)
        for b in self.blocks:
            x = b(x)
        # Mapping to classification output
        x = x.mean(axis=self.mean_axis)
        x = self.linear(x)
        return x
