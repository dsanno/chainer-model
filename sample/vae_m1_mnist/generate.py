import os
import numpy as np
import argparse
from scipy import misc
from chainer import cuda, Variable, serializers
from chainer_trainer.model import VAEModel
from net import MnistM1Net

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',      type=str, required=True,
                    help="input model file path without extension")
parser.add_argument('--output_dir', '-o', type=str, default="generated",
                    help="output directory path")
parser.add_argument('--gpu', '-g',        type=int, default=-1,
                    help="GPU ID (negative value indicates CPU)")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
model = VAEModel(MnistM1Net())
serializers.load_hdf5(args.input + '.model', model)
predictor = model['predictor']

if args.gpu >= 0:
    model.to_gpu(args.gpu)
    xp = cuda.cupy
else:
    model.to_cpu()
    xp = np

image_size = (28, 28)
sample_num = 100
z = xp.random.standard_normal((sample_num, predictor.latent_num)).astype(np.float32)
y = predictor.generate(Variable(z))
y_data = cuda.to_cpu(y.data)

for i in range(sample_num):
    image = np.ones(image_size).astype(np.float32) - y_data[i].reshape(image_size)
    misc.imsave('{}/{}.jpg'.format(args.output_dir, i), image)
