import argparse
import numpy as np

import chainer
import chainer.functions as F
from chainer import cuda
import data
from model import MnistM2Model
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
y_train_category, y_test_category = np.split(mnist['target'], [N])
y_train = np.zeros((y_train_category.shape[0], 10), dtype=np.float32)
y_test = np.zeros((y_test_category.shape[0], 10), dtype=np.float32)
for i in range(y_train_category.shape[0]):
    y_train[i][y_train_category[i]] = 1.0
for i in range(y_test_category.shape[0]):
    y_test[i][y_test_category[i]] = 1.0

if args.input is not None:
    model = Model.load(args.input)
else:
    model = MnistM2Model()

def loss_func((y, mean, var), target):
    return F.mean_squared_error(y, target) - 0.5 * F.sum(1 + var - mean ** 2 - F.exp(var)) / float(y.data.size)

def accuracy_func((y, mean, var), target):
    return F.mean_squared_error(y, target)

Trainer.train(model, (x_train, y_train), x_train, args.iter, x_test=(x_test, y_test), y_test=x_test, batch_size=100, gpu_device=gpu_device, loss_func=loss_func, accuracy_func=accuracy_func)

if args.output is not None:
    model.save(args.output);
