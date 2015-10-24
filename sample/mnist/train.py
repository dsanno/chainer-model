import argparse
import numpy as np

import chainer
import chainer.functions as F
from chainer import cuda
import data
from chainer_trainer.trainer import Trainer
from chainer_trainer.model import Model

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path')
parser.add_argument('--output', '-o', default=None, type=str,
                    help='output model file path')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')

gpu_device = None
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

batch_size = 100

print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])

class MnistModel(Model):
    def __init__(self):
        Model.__init__(self,
            l1 = F.Linear(784, 200),
            l2 = F.Linear(200,200),
            l3 = F.Linear(200,10)
        )

    def forward(self, x_var, train=True):
        h1 = F.dropout(F.relu(self.l1(x_var)),  train=train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=train)
        return self.l3(h2)

if args.input is not None:
    model = Model.load(args.input)
else:
    model = MnistModel()

Trainer.train(model, x_train, y_train, args.iter, x_test=x_test, y_test=y_test, batch_size=100, gpu_device=gpu_device)

if args.output is not None:
    model.save(args.output);
