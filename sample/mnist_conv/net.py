import chainer
import chainer.functions as F
import chainer.links as L

class MnistNet(chainer.Chain):
    def __init__(self):
        super(MnistNet, self).__init__(
            l1 = L.Convolution2D(1, 32, 5, pad=2),
            l2 = L.Convolution2D(32, 64, 5, pad=2),
            l3 = L.Linear(7 * 7 * 64, 1024),
            l4 = L.Linear(1024, 10)
        )

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2)
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)
