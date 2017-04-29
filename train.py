import tensorflow as tf
import os
import numpy as np
import argparse
import style_generator
import utils
import time
import math

from os.path import join, isfile, basename, splitext


def opt_parse():
    parser = argparse.ArgumentParser(description='Train fast neural style',\
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--style_image',default='./image/StarryNight_256.jpg',help='style image path')
    parser.add_argument('--content_dir', default='/home/extra/troutman/tmp3/COCO/train2014_256',\
            help='content images diretory')
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='epoch number')
    parser.add_argument('--alpha', default=1., type=float, help='content loss weight')
    parser.add_argument('--beta', default=1e-4, type=float, help='style loss weight')
    parser.add_argument('--lr', default=1e-2,type=float, help='learning rate')
    parser.add_argument('-TB',default=False, action='store_true',help='Use tensorboard')
    parser.add_argument('--logdir',default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint',default='./checkpoint/', help='checkpoint directory')
    parser.add_argument('--test', default=False, action='store_true',help='test only')
    parser.add_argument('--test_image', default='image/ntu2.png', help='test image')
    parser.add_argument('--train_size', default=512, type=int, help='train image size')

    args = parser.parse_args()
    return args

def _train(args):
    Loss, optimizer, model, content_net = style_generator.build_model(args.style_image,\
            args.alpha, args.beta,args.lr, img_size=args.train_size)
    summary_op = tf.summary.merge_all()

    batch_size = args.batch_size
    img_files = [f for f in os.listdir(args.content_dir) if isfile(join(args.content_dir, f))]
    nb_batch = int(math.ceil((len(img_files)+0.)/batch_size))
    data_batch = utils.coco_input(args.content_dir, batch_size, img_size=args.train_size)

    init = tf.global_variables_initializer()
    
    ### log settings
    if args.TB:
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        writer = tf.summary.FileWriter(args.logdir, graph=tf.get_default_graph())
    
    ### saver settings
    saver = tf.train.Saver()
    if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

    # XXX
    #style_name = splitext(basename(args.style_image))[0]
    #saver_model_path = join(args.checkpoint, style_name) + '/'
    #if not os.path.exists(saver_model_path):
    #        os.makedirs(saver_model_path)

    print('start training')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for ep in xrange(args.epoch):
            ep_time = time.time()
            ep_loss = 0.
            for bs in xrange(nb_batch):
                image = sess.run(data_batch)
                _, loss,out,summary = sess.run([optimizer, Loss, model.out,summary_op],\
                        feed_dict={model.x_input:image,content_net.model['input']:image})
                print bs,loss
                ep_loss += loss

                if (bs+1) % 10 == 0:
                    if args.TB:
                        writer.add_summary(summary, ep * nb_batch + bs)
                    print 'save image' # NOTE debug
                    utils.save_rgb('haha.jpg'.format(ep,bs),out[0][np.newaxis,:])
                if (bs+1) % 1000 == 0:
                    saver.save(sess, join(args.checkpoint, 'model'), global_step=model.global_step)

            print('Epoch:{:4} Loss:{:.4e} Time:{:4f}seconds'.format(ep,ep_loss/nb_batch,time.time()-ep_time))

        coord.request_stop()
        coord.join(threads)


def _test(args):
    test_image = utils.load_rgb(args.test_image)
    content_image = tf.placeholder(tf.float32, shape=test_image.shape, name='content_image')
    model = style_generator.StyleGenerator(256,x=content_image)
    model.build_graph()

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.checkpoint)

    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        _pred = sess.run(model.out,feed_dict={model.x_input:test_image})
        utils.save_rgb('test.png',_pred)


def main(args):
    if not args.test:
        _train(args)
    else:
        _test(args)

if __name__ == '__main__':
    args = opt_parse()
    main(args)
