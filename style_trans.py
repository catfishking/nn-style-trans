import numpy as np
import tensorflow as tf
import argparse
import time

from PIL import Image

config = tf.ConfigProto()                                                                                                                                     
config.gpu_options.allow_growth=True
#config.log_device_placement=True
sess = tf.Session(config=config)


HEIGHT=600
WIDTH=800
CHANNEL=3
nb_epoch = 100000
alpha = 1e-8
beta =1.

VGG_MEAN = [103.939, 116.779, 123.68]

def opt_parse():
    ''' args handling'''
    parser = argparse.ArgumentParser(description='Image Style Transform with vgg19 network')
    parser.add_argument('content',help='content image')
    parser.add_argument('style',help='style image')
    parser.add_argument('-o','--output',default='art.jpg',help='output image')
    parser.add_argument('-v','--vgg19',default='vgg19.npy',help='vgg.npy file')

    return parser.parse_args()

def load_rgb(img_path):
    ''' Load RGB image then return a 4d(1,R,G,B) numpy array'''
    im = Image.open(img_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image = np.array(list(im.getdata()),dtype='float32') - VGG_MEAN
    image = image.reshape(1,im.size[1],im.size[0],3)
    return image

def save_rgb(out_path,array):
    ''' Save RGB image'''
    array = array.reshape(array.shape[1:])
    array = array + VGG_MEAN
    array = np.clip(array,0,255).astype('int8')
    im = Image.fromarray(array,'RGB')
    im.save(out_path)


def build_vgg_model(vgg19_path):
    ''' Build vgg19 network from vgg19.npy
    which is from https://github.com/machrisaa/tensorflow-vgg
    '''
    vgg = np.load(vgg19_path).item()

    model = {}
    model['input'] = tf.Variable(tf.random_uniform([1,HEIGHT,WIDTH,CHANNEL],minval=0,maxval=255))
    model['conv1_1'] = build_vgg_conv(vgg,model['input'],'conv1_1')
    model['conv1_2'] = build_vgg_conv(vgg,model['conv1_1'],'conv1_2')
    model['pool1'] = build_vgg_pool(vgg,model['conv1_2'],'pool1')

    model['conv2_1'] = build_vgg_conv(vgg,model['pool1'],'conv2_1')
    model['conv2_2'] = build_vgg_conv(vgg,model['conv2_1'],'conv2_2')
    model['pool2'] = build_vgg_pool(vgg,model['conv2_2'],'pool2')

    model['conv3_1'] = build_vgg_conv(vgg,model['pool2'],'conv3_1')
    model['conv3_2'] = build_vgg_conv(vgg,model['conv3_1'],'conv3_2')
    model['conv3_3'] = build_vgg_conv(vgg,model['conv3_2'],'conv3_3')
    model['conv3_4'] = build_vgg_conv(vgg,model['conv3_3'],'conv3_4')
    model['pool3'] = build_vgg_pool(vgg,model['conv3_4'],'pool3')

    model['conv4_1'] = build_vgg_conv(vgg,model['pool3'],'conv4_1')
    model['conv4_2'] = build_vgg_conv(vgg,model['conv4_1'],'conv4_2')
    model['conv4_3'] = build_vgg_conv(vgg,model['conv4_2'],'conv4_3')
    model['conv4_4'] = build_vgg_conv(vgg,model['conv4_3'],'conv4_4')
    model['pool4'] = build_vgg_pool(vgg,model['conv4_4'],'pool4')

    model['conv5_1'] = build_vgg_conv(vgg,model['pool4'],'conv5_1')
    model['conv5_2'] = build_vgg_conv(vgg,model['conv5_1'],'conv5_2')
    model['conv5_3'] = build_vgg_conv(vgg,model['conv5_2'],'conv5_3')
    model['conv5_4'] = build_vgg_conv(vgg,model['conv5_3'],'conv5_4')
    model['pool5'] = build_vgg_pool(vgg,model['conv5_4'],'pool5')

    return model



def build_vgg_conv(vgg, prev, name):
    with tf.variable_scope(name):
        filt = tf.constant(vgg[name][0],name='filter')
        conv = tf.nn.conv2d(prev,filt,[1,1,1,1],padding='SAME')
        bias = tf.constant(vgg[name][1],name='bias')

        return tf.nn.relu(conv + bias)

def build_vgg_pool(vgg,prev,name):
    return tf.nn.avg_pool(prev,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name=name)


def set_style_loss(model):

    def gram_matrix(conv,N,M):
        matrix = tf.reshape(conv,(M,N))
        return tf.matmul(tf.transpose(matrix), matrix)

    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    loss = 0.

    for l in layers:
        style_layer = sess.run(model[l])
        N = style_layer.shape[3]
        M = style_layer.shape[1]*style_layer.shape[2]

        G = gram_matrix(style_layer,N,M)
        A = gram_matrix(model[l],N,M)

        loss += 1./(4.* M**2 * N**2)* tf.reduce_sum(tf.pow(G-A,2)) * 1./5.

    return loss
    

def set_content_loss(model):
    content_layer = sess.run(model['conv4_2'])
    N = content_layer.shape[3]
    M = content_layer.shape[1]*content_layer.shape[2]

    #loss = 1./(2. * M * N) * tf.reduce_sum(tf.pow((content_layer - model['conv4_2']),2))
    loss = 1./(2.) * tf.reduce_sum(tf.pow((content_layer - model['conv4_2']),2))
    return loss

if __name__ == '__main__':
    args = opt_parse()
    model = build_vgg_model(args.vgg19)
    img_cont = load_rgb(args.content)
    img_sty = load_rgb(args.style)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(model['input'], (img_cont)))
    content_loss = set_content_loss(model)

    sess.run(tf.assign(model['input'],img_sty))
    style_loss = set_style_loss(model)

    
    Loss = alpha*content_loss + beta*style_loss
    #Loss = style_loss
    #Loss = tf.reduce_sum(tf.pow(cont - model['input'],2))

    optimizer = tf.train.AdamOptimizer(1)
    train_step = optimizer.minimize(Loss)
    
    #writer = tf.train.SummaryWriter('tmp/tf_logs',sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(model['input'],img_cont))
    start = time.time()
    for e in range(nb_epoch):
        _,loss = sess.run([train_step,Loss])
        if (e+1) % 100 == 0:
            print('Epoch:{:5d} Loss:{:e} time:{:8.2f}s'.format(e+1,loss,time.time()-start))
            start = time.time()
            img_art = sess.run(model['input'])
            save_rgb(args.output,img_art)
   
