import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class MnistM2Net(chainer.Chain):

    def __init__(self):
        super(MnistM2Net, self).__init__(
            rec1_x = L.Linear(784, 500),
            rec1_y = L.EmbedID(10, 500),
            rec2 = L.Linear(500, 500),
            rec_mean = L.Linear(500, 50),
            rec_var  = L.Linear(500, 50),
            gen1_z = L.Linear(50, 500),
            gen1_y = L.EmbedID(10, 500),
            gen2 = L.Linear(500, 500),
            gen3 = L.Linear(500, 784)
        )

    def __call__(self, (x_var, y_var), train=True):
        xp = cuda.get_array_module(x_var.data)
        h1 = F.relu(self.rec1_x(x_var) + self.rec1_y(y_var))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        g1 = F.relu(self.gen1_z(z) + self.gen1_y(y_var))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return (g3, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        h1 = F.relu(self.rec1_x(x_rec) + self.rec1_y(y_rec))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)), volatile=True)
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)), volatile=True)
        y_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)), volatile=True)
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + F.exp(var_gen) * Variable(rand, volatile=True)
        g1 = F.relu(self.gen1_z(z) + self.gen1_y(y_gen))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return g3
