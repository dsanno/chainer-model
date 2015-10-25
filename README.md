# chainer_trainer

Chainer training tool

## Usage

Add this directory to environment variable $PYTHONPATH

## Samples

### MNIST

```
$ python sample/mnist/train.py
```

#### Options

* `-i`: Input model file path.
* `-o`: Output model file path.
* `-g`: GPU device index. Default is -1(GPU is not used).
* `--iter`: Number of iteration.
