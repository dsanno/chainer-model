import argparse
import numpy as np

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer_trainer.trainer import Trainer
from chainer_trainer.model import VAEModel
import data
from net import MnistM1Net

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')

gpu_device = None
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255

N = 60000
x_train, x_test = np.split(mnist['data'], [N])

model = VAEModel(MnistM1Net())
optimizer = optimizers.Adam()
optimizer.setup(model)

state = {'max_accuracy': 0}
def progress_func(epoch, loss, accuracy, validate_loss, validate_accuracy, test_loss, test_accuracy):
    print 'epoch: {} done'.format(epoch)
    print('train    mean loss={}, accuracy={}'.format(loss, accuracy))
    if validate_loss is not None and validate_accuracy is not None:
        print('validate mean loss={}, accuracy={}'.format(validate_loss, validate_accuracy))
    if test_loss is not None and test_accuracy is not None:
        print('test     mean loss={}, accuracy={}'.format(test_loss, test_accuracy))
    if epoch % 10 == 0:
        serializers.save_hdf5(args.output + '.model', model)
        serializers.save_hdf5(args.output + '.state', optimizer)

Trainer.train(model, x_train, x_train, args.iter, x_test=x_test, y_test=x_test, batch_size=1000,
    gpu_device=gpu_device, optimizer=optimizer, callback=progress_func)
