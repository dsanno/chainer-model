import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            g1    = L.Linear(50, 1024),
            norm1 = L.BatchNormalization(1024),
            g2    = L.Linear(1024, 7 * 7 * 64),
            norm2 = L.BatchNormalization(7 * 7 * 64),
            g3    = L.Deconvolution2D(64, 32, 5, stride=2, pad=2),
            norm3 = L.BatchNormalization(32),
            g4    = L.Deconvolution2D(32, 1, 5, stride=2, pad=1),
        )

    def __call__(self, z, train=True):
        h1 = F.relu(self.norm1(self.g1(z), test=not train))
        h2 = F.reshape(F.relu(self.norm2(self.g2(h1), test=not train)), (z.data.shape[0], 64, 7, 7))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        return F.sigmoid(self.g4(h3))

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            dc1   = L.Convolution2D(1, 32, 5, stride=2, pad=2),
            norm1 = L.BatchNormalization(32),
            dc2   = L.Convolution2D(32, 64, 5, stride=2, pad=2),
            norm2 = L.BatchNormalization(64),
            dc3   = L.Linear(7 * 7 * 64, 1024),
            norm3 = L.BatchNormalization(1024),
            dc4   = L.Linear(1024, 2)
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        return self.dc4(h3)
