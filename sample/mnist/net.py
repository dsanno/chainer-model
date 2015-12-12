import chainer
import chainer.functions as F
import chainer.links as L

class MnistNet(chainer.Chain):
    def __init__(self):
        super(MnistNet, self).__init__(
            l1 = L.Linear(784, 200),
            l2 = L.Linear(200,200),
            l3 = L.Linear(200,10)
        )

    def __call__(self, x_var, train=True):
        h1 = F.dropout(F.relu(self.l1(x_var)), train=train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=train)
        return self.l3(h2)
