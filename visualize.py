import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def _make_labels(n_images, class_num):
    labels = np.arange(0, class_num, dtype=np.int32)
    for _ in range(n_images // class_num):
        labels = np.append(labels, np.arange(0, class_num, dtype=np.int32))
    return labels[:n_images]

def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            labels = _make_labels(n_images, gen.class_num)
            x = gen(z, labels)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image
