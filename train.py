import tensorflow as tf
import os
import numpy as np
import argparse
import style_generator
import utils
import time

from os.path import join, isfile


def opt_parse():
    parser = argparse.ArgumentParser(description='Train fast neural style',\
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--style_image',default='./image/StarryNight_256.jpg',help='style image path')
    parser.add_argument('--content_dir', default='/home/extra/troutman/tmp3/COCO/train2014_256',\
            help='content images diretory')
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--alpha', default=1., type=float, help='content loss weight')
    parser.add_argument('--beta', default=1e-3, type=float, help='style loss weight')
    parser.add_argument('-TB',default=False, action='store_true')
    parser.add_argument('--logdir',default='./log')

    args = parser.parse_args()
    return args

def main(args):
    Loss, optimizer, model, content_net = style_generator.build_model(args.style_image, args.alpha, args.beta)

    batch_size = args.batch_size
    img_files = [f for f in os.listdir(args.content_dir) if isfile(join(args.content_dir, f))]
    nb_batch = len(img_files)/batch_size

    data_batch = utils.coco_input(args.content_dir, batch_size)

    init = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    
    if args.TB:
        writer = tf.summary.FileWriter(args.logdir, graph=tf.get_default_graph())
    print('start training')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for ep in xrange(200):
            ep_time = time.time()
            ep_loss = 0.
            for bs in xrange(nb_batch):
                image = sess.run(data_batch)
                _, loss,out,summary = sess.run([optimizer, Loss, model.out,summary_op],\
                        feed_dict={model.x_input:image,content_net.model['input']:image})
                print bs,loss

                if args.TB:
                    writer.add_summary(summary, ep * nb_batch + bs)

                if bs % 10 == 9:
                    print 'save'
                    utils.save_rgb('haha.jpg'.format(ep,bs),out[0][np.newaxis,:])
            print('Epoch:{:4} Loss:{:f} Time:{:f}seconds'.format(ep,ep_loss/nb_batch,time.time()-ep_time))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    args = opt_parse()
    main(args)
