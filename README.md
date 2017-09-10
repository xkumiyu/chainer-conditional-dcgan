# Chainer Conditional DCGAN

Chainer implementation of Conditional DCGAN.

# Installation

```
$ cp filelist.txt{.example,}
$ pip install -r requirements.txt
```

# Usage

## Training

```
$ python train.py -i <input filelist>
```

## Inference

```
$ python generate.py -l <label> -m <generator model>
```

### Using trained model

```
$ python generate.py -l <label> -m model/mnist-gen.npz
```

mnist-gen.npz is a model that learned [MNIST](http://yann.lecun.com/exdb/mnist/) data with batchsize=100, epoch=20.

# References

* [Conditional GAN](https://arxiv.org/pdf/1411.1784)
* [DCGAN](https://arxiv.org/abs/1511.06434)
