import numpy as np
import six
import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import cuda

class Trainer(object):
    @classmethod
    def train(self, model, x, y, epoch, x_test=None, y_test=None, loss_func=F.softmax_cross_entropy, accuracy_func=F.accuracy, batch_size=100, optimizer=None, gpu_device=None):
        if gpu_device is None:
            model.to_cpu()
        else:
            model.to_gpu(gpu_device)
        if optimizer is None:
            optimizer = optimizers.Adam()
        optimizer.setup(model)
        for i in six.moves.range(1, epoch + 1):
            print 'epoch: {}'.format(i)
            loss, accuracy = Trainer.train_one(model, x, y, batch_size=batch_size, optimizer=optimizer, loss_func=loss_func, accuracy_func=accuracy_func, gpu_device=gpu_device)
            print('train mean loss={}, accuracy={}'.format(loss, accuracy))
            if x_test is not None and y_test is not None:
                loss, accuracy = Trainer.train_one(model, x_test, y_test, batch_size=batch_size, loss_func=loss_func, accuracy_func=accuracy_func, gpu_device=gpu_device)
                print('test mean loss={}, accuracy={}'.format(loss, accuracy))

    @classmethod
    def train_one(self, model, x, y, loss_func, accuracy_func, batch_size, gpu_device, optimizer=None):
        x_is_tuple = isinstance(x, tuple)
        y_is_tuple = isinstance(y, tuple)
        train = optimizer is not None and loss_func is not None
        if x_is_tuple:
            total_size = len(x[0])
            assert all(total_size == len(z) for z in x)
        else:
            total_size = len(x)
        if y_is_tuple:
            assert all(total_size == len(z) for z in y)
        else:
            assert total_size == len(y)
        if gpu_device is None:
            xp = np
        else:
            xp = cuda.cupy
        perm = np.random.permutation(total_size)
        sum_loss = 0
        sum_accuracy = 0
        for i in six.moves.range(0, total_size, batch_size):
            to_batch_variable = lambda data: chainer.Variable(xp.asarray(data[perm[i:i + batch_size]]))
            if x_is_tuple:
                x_batch = tuple(map(to_batch_variable, x))
            else:
                x_batch = to_batch_variable(x)
            if y_is_tuple:
                y_batch = tuple(map(to_batch_variable, y))
            else:
                y_batch = to_batch_variable(y)
            if train:
                optimizer.zero_grads()
            y_out = model.forward(x_batch, train=train)
            loss = loss_func(y_out, y_batch)
            if train:
                loss.backward()
                optimizer.update()
            sum_loss += float(loss.data) * batch_size
            sum_accuracy += float(accuracy_func(y_out, y_batch).data) * batch_size
        return (sum_loss / total_size, sum_accuracy / total_size)
