import argparse
import os

import chainer
from chainer import Variable
import numpy as np
from PIL import Image

from net import Generator


def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--label', '-l', type=int, default=0,
                        help='Label to generate image')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output')
    parser.add_argument('--model', '-m', default='result/gen.npz',
                        help='Snapshot of Generator')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    args = parser.parse_args()

    gen = Generator(n_hidden=args.n_hidden)
    chainer.serializers.load_npz(args.model, gen)
    if args.gpu >= 0:
        gen.to_gpu(args.gpu)

    np.random.seed(args.seed)
    z = Variable(np.asarray(gen.make_hidden(1)))
    with chainer.using_config('train', False):
        x = gen(z, [args.label])[0]

    x = np.asarray(np.clip(x.data * 255, 0.0, 255.0), dtype=np.uint8).transpose(1, 2, 0)

    path = os.path.join(args.out, '{}.png'.format(args.label))
    Image.fromarray(x).save(path)


if __name__ == '__main__':
    main()
