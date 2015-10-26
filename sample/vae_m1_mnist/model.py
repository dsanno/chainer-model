import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import Variable
from chainer_trainer.model import Model

class MnistM1Model(Model):
    def __init__(self):
        Model.__init__(self,
            rec1 = F.Linear(784, 500),
            rec2 = F.Linear(500, 500),
            rec_mean = F.Linear(500, 50),
            rec_var  = F.Linear(500, 50),
            gen1 = F.Linear(50,  500),
            gen2 = F.Linear(500, 500),
            gen3 = F.Linear(500, 784)
        )

    def forward(self, x_var, train=True):
        xp = cuda.get_array_module(x_var.data)
        h1 = F.relu(self.rec1(x_var))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand)
        g1 = F.relu(self.gen1(z))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return (g3, mean, var)

    def generate(self, z):
        g1 = F.relu(self.gen1(z))
        g2 = F.relu(self.gen2(g1))
        return F.sigmoid(self.gen3(g2))
