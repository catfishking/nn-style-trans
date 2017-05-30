import tensorflow as tf
import numpy as np
import argparse
import os
import math
import time


from os.path import isfile, join

import ops
import utils


def opt_parse():
    parser = argparse.ArgumentParser(description="C-Cycle GAN style transfer")
    parser.add_argument('--style_dir',default='/tmp3/troutman/WikiArt/wikiart/ha',
            help='style image path')
    parser.add_argument('--content_dir', default='/home/extra/troutman/tmp3/COCO/train2014_256',
            help='content images diretory')
    parser.add_argument('--batch_size', default=5, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='epoch number')
    parser.add_argument('--lr', default=1e-2,type=float, help='learning rate')
    parser.add_argument('-TB',default=False, action='store_true',help='Use tensorboard')
    parser.add_argument('--logdir',default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint',default='./checkpoint/cycle', help='checkpoint directory')
    parser.add_argument('--test', default=False, action='store_true',help='test only')
    parser.add_argument('--test_image', default='image/ntu2.png', help='test image')
    return parser.parse_args()
    
class CycleGan(object):
    def __init__(self,image_size=256):
        self.image_size = image_size


    def build_graph(self):

        self.style_image_x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
        self.real_image_x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
        self.style_image = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
        self.real_image = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])

        # realistic to style
        Gx = self.build_G('X', image=self.real_image_x, style_image = self.style_image, reuse=False)
        Dx_fake = self.build_D('X', image=Gx, style_image = self.style_image, reuse=False)
        Gy_x = self.build_G('X', image=Gx, style_image = self.real_image, reuse=True)
        Dx_real = self.build_D('X', image=self.style_image_x, style_image = self.style_image, reuse=True)

        # style to realistic
        Gy = self.build_G('X', image=self.style_image_x, style_image = self.real_image, reuse=True)
        Dy_fake = self.build_D('X', image=Gy, style_image=self.real_image, reuse=True)
        Gx_y = self.build_G('X', image=Gy, style_image=self.style_image, reuse=True)
        Dy_real = self.build_D('X', image=self.real_image_x, style_image=self.real_image, reuse=True)

        # WGAN loss
        self.Dx_loss = tf.reduce_mean(Dx_real) - tf.reduce_mean(Dx_fake)
        self.Gx_loss = -tf.reduce_mean(Dx_fake)

        self.Dy_loss = tf.reduce_mean(Dy_real) - tf.reduce_mean(Dy_fake)
        self.Gy_loss = -tf.reduce_mean(Dy_fake)

        # cycle consistent loss
        self.cycle_loss = tf.reduce_mean(tf.abs(Gx_y - self.style_image_x)) \
                + tf.reduce_mean(tf.abs(Gy_x - self.real_image_x))
        self.G_loss = self.Gy_loss + 10 * self.cycle_loss
        self.D_loss = self.Dx_loss + self.Dy_loss

        # get G D variables
        t_vars = tf.trainable_variables()
        self.d_var = [ var for var in t_vars if 'Discriminator_' in var.name ]
        self.g_var = [ var for var in t_vars if 'Generator_' in var.name ]

        # compute gradient
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4)\
                .minimize(-self.D_loss, var_list=self.d_var)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4)\
                .minimize(self.G_loss, var_list=self.g_var)

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in self.d_var]
        self.Gx_out = Gx
        self.Gy_out = Gy
        #clip_Dy = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_var]

        # summary
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('D_loss', self.D_loss)
        self.summary_op = tf.summary.merge_all()


    def train(self, ART_dir, COCO_dir, batch_size=10, TB=False, checkpoint='./checkpoint/'):
        # build content data generator
        content_files = [join(COCO_dir,f) for f in os.listdir(COCO_dir) if isfile(join(COCO_dir, f))]
        content_data_generator = utils.image_generator(content_files, batch_size=batch_size)

        # build style data generator
        artists_list = ['vincent','monet','cezanne','katsu']
        style_data_generator = []
        for art in artists_list:
            artist_dir = ART_dir + '/images_256_'+art
            art_files = [join(artist_dir, f) for f in os.listdir(artist_dir) if isfile(join(artist_dir,f))]
            style_data_generator.append(utils.image_generator(art_files, batch_size=batch_size))

        nb_batch = int(math.ceil((len(content_files)+0.)/batch_size))

        ### log settings
        if args.TB:
            if not os.path.exists(args.logdir):
                os.makedirs(args.logdir)
            writer = tf.summary.FileWriter(args.logdir, graph=tf.get_default_graph())
        
        ### saver settings
        saver = tf.train.Saver()
        if not os.path.exists(checkpoint):
                os.makedirs(checkpoint)

        init = tf.global_variables_initializer()
        print('start training')
        with tf.Session() as sess:
            sess.run(init)
            for ep in xrange(args.epoch):
                ep_time = time.time()
                ep_loss = 0.
                for bs in xrange(nb_batch):
                    for art in range(len(artists_list)):
                        for _ in range(10): # n critic
                            # Read data
                            content_images = next(content_data_generator)
                            style_images = next(style_data_generator[art])

                            content_img = np.repeat(content_images[0:1], batch_size, axis=0)
                            style_img = np.repeat(style_images[0:1], batch_size, axis=0)

                            _, _, D_loss = sess.run(
                                    [self.D_solver, self.clip_D, self.D_loss],
                                    feed_dict={self.style_image_x:style_images, self.real_image_x:content_images,
                                        self.style_image:style_img, self.real_image:content_img})
                        _, G_loss, Cycle_loss, Gx_out, Gy_out = sess.run(
                                    [self.G_solver, self.G_loss, self.cycle_loss, self.Gx_out, self.Gy_out],
                                    feed_dict={self.style_image_x:style_images, self.real_image_x:content_images,
                                        self.style_image:style_img, self.real_image:content_img})

                        print ('Epoch:{:5}  Step:{:5}  D_loss{:f} G_loss:{:f} Cycle_loss:{:f}'.format(
                            ep, bs, D_loss, G_loss, Cycle_loss))

                    # save log
                    if (bs+1) % 10 == 0:
                        # save Tensorbroad log
                        if args.TB:
                            writer.add_summary(summary, ep * nb_batch + bs)
                        print 'save image' # NOTE debug
                        utils.save_rgb('haha.jpg'.format(ep,bs),Gx_out[0][np.newaxis,:])
                        utils.save_rgb('haha2.jpg'.format(ep,bs),Gy_out[0][np.newaxis,:])
                        utils.save_rgb('haha_in.jpg'.format(ep,bs),content_img[0][np.newaxis,:])
                        utils.save_rgb('haha2_in.jpg'.format(ep,bs),style_img[0][np.newaxis,:])
                    if (bs+1) % 1000 == 0:
                        saver.save(sess, join(args.checkpoint, 'model'), global_step=ep*nb_batch+bs)
                    


    def build_G(self, name, image=None, style_image=None, reuse=False):
        with tf.variable_scope('Generator_'+name, reuse=reuse):
            if image is None:
                self.x_input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
                self.y_input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
                x = tf.concat(self.x_input, self.y_input, 2)
            elif image is not None and style_image is not None:
                x = tf.concat([image, style_image], 3)

            # filter_size:9, nb_filter: 3->32, step:1
            x = ops.conv_layer('first', x, 9, 6, 32, 1)
            # filter_size:3, nb_filter: 32->64, step:1
            x = ops.conv_layer('dsample_1', x, 3, 32, 64, 2)
            # filter_size:3, nb_filter:64->128, step:1
            x = ops.conv_layer('dsample_2', x, 3, 64, 128, 2)

            for i in range(5): # five residual blocks
                name = ('res_block_{}'.format(i))
                # filter_size:3, nb_filter:128->128, step:1
                x = ops.residual(name, x, 128, 128, 1)

            x = ops.upsample_nearest_neighbor('usample_1',x,3,128,64,2)
            x = ops.upsample_nearest_neighbor('usample_2',x,3,64,32,2)
            x = ops.conv_layer('last',x,9,32,3,1,activation=False)
            out = tf.nn.tanh(x) * 150 + 255./2

            return out


    def build_D(self, name, image=None, style_image=None, reuse=False):
        with tf.variable_scope('Discriminator_'+name, reuse=reuse):
            if image is None:
                self.d_input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 6])
            elif image is not None and style_image is not None:
                self.d_input = tf.concat([image, style_image], 3)
            
            # filter_size:4, nb_filter: 3->64, step:2
            x = ops.conv_layer('conv_1',self.d_input, 4, 6, 64, 2, activation=True, leakiness=0.2)
            # filter_size:4, nb_filter: 64->128, step:2
            x = ops.conv_layer('conv_2', x, 4, 64, 128, 2, activation=True, leakiness=0.2)
            # filter_size:4, nb_filter: 64->128, step:2
            x = ops.conv_layer('conv_3', x, 4, 128, 256, 2, activation=True, leakiness=0.2)
            x = ops.conv_layer('conv_4', x, 4, 256, 512, 2, activation=True, leakiness=0.2)
            x = ops.conv_layer('conv_5', x, 4, 512, 512, 2, activation=True, leakiness=0.2)
            x = ops.conv_layer('conv_6', x, 4, 512, 1, 4, activation=True, leakiness=0.2)
            
            out = x
            #self.label = tf.placeholder(tf.float32, [None])
            return out


    def build_G_AdaIN(self, name, input_size=256, image=None, style_image=None):
        with tf.variable_scope('Generator_encoder_'+name):
            if image is None:
                x_input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
            else:
                x_input = image

            # filter_size:9, nb_filter: 3->32, step:1
            x = ops.conv_layer('first', x_input, 9, 3, 32, 1)
            # filter_size:3, nb_filter: 32->64, step:1
            x = ops.conv_layer('dsample_1', x, 3, 32, 64, 2)
            # filter_size:3, nb_filter:64->128, step:1
            x = ops.conv_layer('dsample_2', x, 3, 64, 128, 2)


        with tf.variable_scope('Generator_encoder_c_'+name, reuse=True):
            if image is None:
                y_input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
            else:
                y_input = style_image

            # filter_size:9, nb_filter: 3->32, step:1
            y = ops.conv_layer('first', y_input, 9, 3, 32, 1)
            # filter_size:3, nb_filter: 32->64, step:1
            y = ops.conv_layer('dsample_1', x, 3, 32, 64, 2)
            # filter_size:3, nb_filter:64->128, step:1
            y = ops.conv_layer('dsample_2', x, 3, 64, 128, 2)

        with tf.variable_scope('Generator_decoder_'+name):
            for i in range(5): # five residual blocks
                name = ('res_block_{}'.format(i))
                # filter_size:3, nb_filter:128->128, step:1
                x = ops.residual(name, x, 128, 128, 1, norm='AdaIN', y=y)

            x = ops.upsample_nearest_neighbor('usample_1',x,3,128,64,2)
            x = ops.upsample_nearest_neighbor('usample_2',x,3,64,32,2)
            x = ops.conv_layer('last',x,9,32,3,1,activation=False)
            out = tf.nn.tanh(x) * 150 + 255./2

            return out


def train_GAN(args):
    gan = CycleGan()
    gan.build_graph()
    gan.train(args.style_dir, args.content_dir, args.batch_size, args.TB, args.checkpoint)



if __name__ == '__main__':
    args = opt_parse()
    train_GAN(args)

