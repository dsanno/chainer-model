import argparse
import numpy as np
import os
from PIL import Image

import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import data
from net import Generator, Discriminator

parser = argparse.ArgumentParser(description='Chainer training example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')
parser.add_argument('--out_image_dir', default=None, type=str,
                    help='output directory to output images')
args = parser.parse_args()

gen_model = Generator()
optimizer_gen = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_gen.setup(gen_model)
optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
dis_model = Discriminator()
optimizer_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_dis.setup(dis_model)
optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

if args.input != None:
    serializers.load_hdf5(args.input + '.gen.model', gen_model)
    serializers.load_hdf5(args.input + '.gen.state', optimizer_gen)
    serializers.load_hdf5(args.input + '.dis.model', dis_model)
    serializers.load_hdf5(args.input + '.dis.state', optimizer_dis)

if args.out_image_dir != None:
    if not os.path.exists(args.out_image_dir):
        try:
            os.mkdir(args.out_image_dir)
        except:
            print 'cannot make directory {}'.format(args.out_image_dir)
            exit()
    elif not os.path.isdir(args.out_image_dir):
        print 'file path {} exists but is not directory'.format(args.out_image_dir)
        exit()

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255

N = 60000
x_train, x_test = np.split(mnist['data'], [N])
LATENT_SIZE = 50
BATCH_SIZE = 100

def train_gen(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = xp
    else:
        xp = cuda.cupy
    z = Variable(xp.random.uniform(-1, 1, (batch_size, LATENT_SIZE)).astype(np.float32))
    x = gen(z)
    y1 = dis(x)
    loss_gen = F.softmax_cross_entropy(y1, Variable(xp.zeros(batch_size).astype(np.int32)))
    loss_dis = F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size).astype(np.int32)))
    optimizer_gen.zero_grads()
    loss_gen.backward()
    optimizer_gen.update()
    return loss_gen.data

def train_dis(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = xp
    else:
        xp = cuda.cupy
    z = Variable(xp.random.uniform(-1, 1, (batch_size, LATENT_SIZE)).astype(np.float32))
    x = gen(z)
    y1 = dis(x)
    loss_dis = F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size).astype(np.int32)))
    x2 = Variable(xp.asarray(np.reshape(x_batch, (batch_size, 1, 28, 28))))
    y2 = dis(x2)
    loss_dis += F.softmax_cross_entropy(y2, Variable(xp.zeros(batch_size).astype(np.int32)))
    optimizer_dis.zero_grads()
    loss_dis.backward()
    optimizer_dis.update()
    return loss_dis.data

def train_one(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device):
    batch_size = len(x_batch)
    if gpu_device == None:
        xp = xp
    else:
        xp = cuda.cupy
    # train generator
    z = Variable(xp.random.uniform(-1, 1, (batch_size, LATENT_SIZE)).astype(np.float32))
    x = gen(z)
    y1 = dis(x)
    loss_gen = F.softmax_cross_entropy(y1, Variable(xp.zeros(batch_size).astype(np.int32)))
    loss_dis = F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size).astype(np.int32)))
    # train discriminator
    x2 = Variable(xp.asarray(np.reshape(x_batch, (batch_size, 1, 28, 28))))
    y2 = dis(x2)
    loss_dis += F.softmax_cross_entropy(y2, Variable(xp.zeros(batch_size).astype(np.int32)))

    optimizer_gen.zero_grads()
    loss_gen.backward()
    optimizer_gen.update()

    optimizer_dis.zero_grads()
    loss_dis.backward()
    optimizer_dis.update()

    return (loss_gen.data, loss_dis.data)

def train(gen, dis, optimizer_gen, optimizer_dis, x_train, epoch_num, gpu_device=None, out_image_dir=None):
    if gpu_device == None:
        gen.to_cpu()
        dis.to_cpu()
        xp = np
    else:
        gen.to_gpu(gpu_device)
        dis.to_gpu(gpu_device)
        xp = cuda.cupy
    out_image_len = 20
    z_out_image =  Variable(xp.random.uniform(-1, 1, (out_image_len, LATENT_SIZE)).astype(np.float32))
    for epoch in xrange(1, epoch_num + 1):
        x_size = len(x_train)
        perm = np.random.permutation(x_size)
        sum_loss_gen = 0
        sum_loss_dis = 0
        for i in xrange(0, x_size, BATCH_SIZE):
            x_batch = x_train[perm[i:i + BATCH_SIZE]]
            loss_dis = train_dis(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device)
            sum_loss_dis += float(loss_dis)
            loss_gen = train_gen(gen, dis, optimizer_gen, optimizer_dis, x_batch, gpu_device)
            sum_loss_gen += float(loss_gen)
        print 'epoch: {} done'.format(epoch)
        print('gen loss={}'.format(sum_loss_gen / x_size))
        print('dis loss={}'.format(sum_loss_dis / x_size))
        serializers.save_hdf5(args.output + '.gen.model', gen)
        serializers.save_hdf5(args.output + '.gen.state', optimizer_gen)
        serializers.save_hdf5(args.output + '.dis.model', dis)
        serializers.save_hdf5(args.output + '.dis.state', optimizer_dis)
        if out_image_dir != None:
            data_array = gen(z_out_image, train=False).data
            for i, data in enumerate(data_array):
                image = Image.fromarray((cuda.to_cpu(data) * 256).astype(np.uint8).reshape(data.shape[1:3]))
                image.save('{0}/{1:03d}_{2:03d}.png'.format(out_image_dir, epoch, i))

train(gen_model, dis_model, optimizer_gen, optimizer_dis, x_train, args.iter, gpu_device=gpu_device, out_image_dir=args.out_image_dir)
