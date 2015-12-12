import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class MnistM1Net(chainer.Chain):

    latent_num = 50

    def __init__(self):
        super(MnistM1Net, self).__init__(
            rec1 = L.Linear(784, 500),
            rec2 = L.Linear(500, 500),
            rec_mean = L.Linear(500, self.latent_num),
            rec_var  = L.Linear(500, self.latent_num),
            gen1 = L.Linear(self.latent_num, 500),
            gen2 = L.Linear(500, 500),
            gen3 = L.Linear(500, 784)
        )

    def __call__(self, x_var, train=True):
        xp = cuda.get_array_module(x_var.data)
        h1 = F.relu(self.rec1(x_var))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.relu(self.gen1(z))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return (g3, mean, var)

    def generate(self, z):
        g1 = F.relu(self.gen1(z))
        g2 = F.relu(self.gen2(g1))
        return F.sigmoid(self.gen3(g2))
