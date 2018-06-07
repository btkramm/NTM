# coding=utf-8

"""If Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np

# from ntm.aio import EncapsulatedNTM


def test(num_batches, batch_size):
  for batch_num in range(num_batches):
    seq_len, seq_width = 3, 8

    seq_tensor    = np.empty((batch_size, 3, seq_width))
    output_tensor = np.empty((batch_size, 1, seq_width))

    for instance in range(batch_size):
      options = np.random.binomial(1, 0.5, (2, seq_width))

      if np.random.randint(2) == 1:
        condition = np.ones((1, seq_width))
        answer    = options[0]
      else:
        condition = np.zeros((1, seq_width))
        answer    = options[1]

      seq_tensor[instance]       = np.concatenate((condition, options))
      output_tensor[instance][0] = answer

    seq_tensor    = torch.from_numpy(seq_tensor)
    output_tensor = torch.from_numpy(output_tensor)

    # print(seq_tensor.shape, output_tensor.shape)

    input_tensor = Variable(torch.zeros(batch_size, 4, seq_width + 1))
    input_tensor[:, :3, :seq_width] = seq_tensor
    input_tensor[:, 3, seq_width]   = 1.0

    # print(seq_tensor[0], output_tensor[0])
    # print(seq_tensor[1], output_tensor[1])

    # yield batch_num+1, input_tensor.float(), output_tensor.float()


@attrs
class CopyTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=8, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)


#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class CopyTaskModelTraining(object):
    params = attrib(default=Factory(CopyTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 1, self.params.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_min_len, self.params.sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)


if __name__ == '__main__':
  test(1, 16)

  # dataloader(1, 8, 8, 3, 3)
