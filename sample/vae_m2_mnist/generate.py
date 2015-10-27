import os
import numpy as np
import argparse
from chainer_trainer.model import Model
from scipy import misc
from chainer import cuda, Variable
import data

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',      type=str, required=True,
                    help="input model file path")
parser.add_argument('--output_dir', '-o', type=str, default="generated",
                    help="output directory path")
parser.add_argument('--gpu', '-g',        type=int, default=-1,
                    help="GPU ID (negative value indicates CPU)")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
model = Model.load(args.input)
if args.gpu >= 0:
    model.to_gpu(args.gpu)
    xp = cuda.cupy
else:
    model.to_cpu()
    xp = np

print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
_, x_test = np.split(mnist['data'],   [N])
_, y_test_category = np.split(mnist['target'], [N])
y_test = np.zeros((y_test_category.shape[0], 10), dtype=np.float32)
for i in range(y_test_category.shape[0]):
    y_test[i][y_test_category[i]] = 1.0

perm = np.random.permutation(len(y_test))
sample_num = 100
category_num = 10
W = 28
H = 28
y_gen = Variable(xp.identity(category_num).astype(np.float32))
for i in range(sample_num):
    j = perm[i]
    x = Variable(xp.asarray(x_test[j:j + 1]))
    y_rec = Variable(xp.asarray(y_test[j:j + 1]))
    y = model.generate(x, y_rec, y_gen)
    y_data = cuda.to_cpu(y.data)
    x_data = cuda.to_cpu(x.data)
    image = 1.0 - np.vstack((x_data, y_data)).reshape(category_num + 1, H, W).swapaxes(0, 1).reshape(H, (category_num + 1) * W)
    misc.imsave('{}/{}.jpg'.format(args.output_dir, i), image)
