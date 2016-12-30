import numpy as np
import tensorflow as tf
import argparse

from PIL import Image

config = tf.ConfigProto()                                                                                                                                     
config.gpu_options.allow_growth=True
#config.log_device_placement=True
sess = tf.Session(config=config)
HEIGHT=600
WIDTH=600
CHANNEL=3

def opt_parse():
    ''' args handling'''
    parser = argparse.ArgumentParser(description='Image Style Transform with vgg19 network')
    parser.add_argument('content',help='content image')
    parser.add_argument('style',help='style image')
    parser.add_argument('output',help='output image')
    parser.add_argument('-v','--vgg19',default='vgg19.npy',help='vgg.npy file')

    return parser.parse_args()

def build_vgg_model(vgg19_path):
    ''' Build vgg19 network from vgg19.npy
    which is from https://github.com/machrisaa/tensorflow-vgg
    '''
    vgg = np.load(vgg19_path).item()

    model = {}
    model['input'] = tf.Variable(tf.zeros([1,HEIGHT,WIDTH,CHANNEL]))
    model['conv1_1'] = build_vgg_conv(vgg,model['input'],'conv1_1')
    model['conv1_2'] = build_vgg_conv(vgg,model['conv1_1'],'conv1_2')
    model['pool1'] = build_vgg_pool(vgg,model['conv1_2'],'pool1')

    model['conv2_1'] = build_vgg_conv(vgg,model['pool1'],'conv2_1')
    model['conv2_2'] = build_vgg_conv(vgg,model['conv2_1'],'conv2_2')
    model['pool2'] = build_vgg_pool(vgg,model['conv2_2'],'pool2')

    model['conv3_1'] = build_vgg_conv(vgg,model['pool2'],'conv3_1')
    model['conv3_2'] = build_vgg_conv(vgg,model['conv3_1'],'conv3_2')
    model['conv3_3'] = build_vgg_conv(vgg,model['conv3_2'],'conv3_3')
    model['pool3'] = build_vgg_pool(vgg,model['conv3_3'],'pool3')

    model['conv4_1'] = build_vgg_conv(vgg,model['pool3'],'conv4_1')
    model['conv4_2'] = build_vgg_conv(vgg,model['conv4_1'],'conv4_2')
    model['conv4_3'] = build_vgg_conv(vgg,model['conv4_2'],'conv4_3')
    model['pool4'] = build_vgg_pool(vgg,model['conv4_3'],'pool4')

    model['conv5_1'] = build_vgg_conv(vgg,model['pool4'],'conv5_1')
    model['conv5_2'] = build_vgg_conv(vgg,model['conv5_1'],'conv5_2')
    model['conv5_3'] = build_vgg_conv(vgg,model['conv5_2'],'conv5_3')
    model['pool5'] = build_vgg_pool(vgg,model['conv5_3'],'pool5')

    return model



def build_vgg_conv(vgg, prev, name):
    with tf.variable_scope(name):
        filt = tf.constant(vgg[name][0],name='filter')
        conv = tf.nn.conv2d(prev,filt,[1,1,1,1],padding='SAME')
        bias = tf.constant(vgg[name][1],name='bias')

        return tf.nn.relu(conv + bias)

def build_vgg_pool(vgg,prev,name):
    return tf.nn.avg_pool(prev,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name=name)

if __name__ == '__main__':
    args = opt_parse()
    model = build_vgg_model(args.vgg19)
    sess = tf.InteractiveSession()
    tf.summary.FileWriter('tmp/test.log',sess.graph)

