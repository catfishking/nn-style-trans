import tensorflow as tf
import numpy as np


def stride_arr(stride):
    '''Map a stride scalar to the stride array for tf.nn.conv2d.'''
    return [1, stride, stride, 1]


def out_shape( x, stride, out_dim):
    ''' Build out_shape array for tf.nn.conv2d_transpose '''
    # use tf.shape(x)[0] to obtain batch size
    return [ tf.shape(x)[0], x.get_shape()[1].value*stride,\
            x.get_shape()[2].value*stride,out_dim]


def conv_layer(name,x, filter_size, in_filter, out_filter, stride,\
                activation=True, leakiness=0.0, norm='IN', y=None):
    ''' convolutional layer
        Arguments:
            activation: use ReLU after convulotion
            leakiness: leakiness of leakyReLU
            norm: normalization method. BN|IN|AdaIN
    '''

    with tf.variable_scope(name):
        x = conv('conv', x, filter_size, in_filter, out_filter, stride_arr(stride) )

        # which norm method
        if norm == 'BN':
            x = batch_norm('bn', x)
        elif norm == 'IN':
            x = instance_norm('in',x)
        elif norm == 'AdaIN':
            x = adaptive_instance_norm('AdaIN', x, y)

        if activation:
            x = relu(x, leakiness)
        return x


def upsample( name, x, filter_size,in_filter, out_filter, stride):
    with tf.variable_scope(name):
        x = conv_trans('conv_trans', x, filter_size, in_filter, out_filter,\
                stride_arr(2),out_shape(x, stride, out_filter))
        x = instance_norm('bn',x)
        return x


def upsample_nearest_neighbor(name,x, filter_size,in_filter,out_filter,stride):
    with tf.variable_scope(name):
        _, height, width, _ = [i.value for i in x.get_shape()]
        x = tf.image.resize_nearest_neighbor(x, [stride*height, stride*width])
        x = conv_layer('conv', x, filter_size, in_filter, out_filter, 1)
        return x


def residual(name, x,in_filter,out_filter,stride, leakiness=0., norm='IN', y=None):
    ''' residual block
        Arguments:
            leakiness: leakiness of leakyReLU
            norm: normalization method. BN|IN|AdaIN
    '''
    orig_x = x
    with tf.variable_scope(name):
        x = conv_layer('sub1',x,3,in_filter,out_filter,stride,\
                leakiness=leakiness, norm=norm, y=y)
        x = conv_layer('sub2',x,3,in_filter,out_filter,stride, activation=False,\
                leakiness=leakiness, norm=norm, y=y)
        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],\
                                [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x
        return x


# NOTE: use batch_norm in tf layers
def batch_norm(name,x, mode):
    ''' mode = train | test '''
    if mode == 'train':
        BN = tf.contrib.layers.batch_norm(x,center=True,scale=True,\
            epsilon=1e-5,is_training=True,scope=name)
    else:
        BN = tf.contrib.layers.batch_norm(x,center=True,scale=True,\
                epsilon=1e-5,is_training=False,scope=name)
    return BN


def instance_norm(name,x):
    with tf.variable_scope(name):
        channels = x.get_shape()[3].value
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(x, [1,2], keep_dims=True)
        shift = tf.get_variable('shift', var_shape, tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', var_shape, tf.float32, initializer=tf.ones_initializer())
        epsilon = 1e-5
        normalized = (x-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift

def adaptive_instance_norm(name,x,y):
    ''' AdaIN from Arbitrary Style Transfer in Real-time with Adaptive \
            Instance Normalization'''
    with tf.variable_scope(name):
        channels = x.get_shape()[3].value
        var_shape = [channels]
        mu_x, sigma_sq_x = tf.nn.moments(x, [1,2], keep_dims=True)
        mu_y, sigma_sq_y = tf.nn.moments(y, [1,2], keep_dims=True)
        epsilon = 1e-5
        AdaIN = sigma_sq_y**(.5) * (x - mu_x)/(sigma_sq_x + epsilon)**(.5) + mu_y
        return AdaIN
        

def conv(name,x,filter_size,in_filters,out_filters,stride):
    ''' Convolution '''
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        # NOTE not use shared variable
        #kernel = tf.get_variable('W',[filter_size,filter_size,in_filters,out_filters],\
        #        tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        kernel = tf.get_variable('W', [filter_size, filter_size, in_filters, out_filters],\
                tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x,kernel,stride,padding='SAME')


def conv_trans(name, x, filter_size, in_filters, out_filters, stride, out_shape):
    ''' transpose convolution '''
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('W', [filter_size, filter_size, out_filters, out_filters],\
                tf.float32, initializer=tf.random_normal(stddev=np.sqrt(2.0/n)))
        x = tf.nn.conv2d_transpose(x, kernel, out_shape, stride, padding='SAME')

        # NOTE after conv2d_transpose, dim info loss... use this hack keep static dim info
        x = tf.reshape(x,out_shape)
        return x


# TODO pass relu leakiness argvs
def relu(x,leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu' )


