import chainer
import chainer.functions as F
from chainer import Variable


class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        images = [batch[i][0] for i in range(batchsize)]
        labels = [batch[i][1] for i in range(batchsize)]

        image_size = (32, 32)
        x_real = F.resize_images(
            Variable(self.converter(images, self.device)) / 255., image_size)
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis

        # print('x_real', x_real.shape)
        y_real = dis(x_real, labels)
        # print('y_real', y_real.shape)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        # print('z', z.shape)
        x_fake = gen(z, labels)
        # print('x_fake', x_fake.shape)
        y_fake = dis(x_fake, labels)
        # print('y_fake', y_fake.shape)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
