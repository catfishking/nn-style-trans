import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


class VGG19:
    ''' Build vgg19 network from vgg19.npy
    which is from https://github.com/machrisaa/tensorflow-vgg
    '''
    def __init__(self,vgg19_path,x=None,reuse=False,HEIGHT=600,WIDTH=800,CHANNEL=3):
        vgg = np.load(vgg19_path).item()
        
        with tf.variable_scope('VGG19',reuse=reuse):
            self.model = {}
            if x == None:
                self.model['input'] = tf.Variable(tf.random_uniform([1,HEIGHT,WIDTH,CHANNEL],minval=0,maxval=255))
            else:
                self.model['input'] = x
            self.model['conv1_1'] = self.build_vgg_conv(vgg,self.model['input'],'conv1_1')
            self.model['conv1_2'] = self.build_vgg_conv(vgg,self.model['conv1_1'],'conv1_2')
            self.model['pool1'] = self.build_vgg_pool(vgg,self.model['conv1_2'],'pool1')

            self.model['conv2_1'] = self.build_vgg_conv(vgg,self.model['pool1'],'conv2_1')
            self.model['conv2_2'] = self.build_vgg_conv(vgg,self.model['conv2_1'],'conv2_2')
            self.model['pool2'] = self.build_vgg_pool(vgg,self.model['conv2_2'],'pool2')

            self.model['conv3_1'] = self.build_vgg_conv(vgg,self.model['pool2'],'conv3_1')
            self.model['conv3_2'] = self.build_vgg_conv(vgg,self.model['conv3_1'],'conv3_2')
            self.model['conv3_3'] = self.build_vgg_conv(vgg,self.model['conv3_2'],'conv3_3')
            self.model['conv3_4'] = self.build_vgg_conv(vgg,self.model['conv3_3'],'conv3_4')
            self.model['pool3'] = self.build_vgg_pool(vgg,self.model['conv3_4'],'pool3')

            self.model['conv4_1'] = self.build_vgg_conv(vgg,self.model['pool3'],'conv4_1')
            self.model['conv4_2'] = self.build_vgg_conv(vgg,self.model['conv4_1'],'conv4_2')
            self.model['conv4_3'] = self.build_vgg_conv(vgg,self.model['conv4_2'],'conv4_3')
            self.model['conv4_4'] = self.build_vgg_conv(vgg,self.model['conv4_3'],'conv4_4')
            self.model['pool4'] = self.build_vgg_pool(vgg,self.model['conv4_4'],'pool4')

            self.model['conv5_1'] = self.build_vgg_conv(vgg,self.model['pool4'],'conv5_1')
            self.model['conv5_2'] = self.build_vgg_conv(vgg,self.model['conv5_1'],'conv5_2')
            self.model['conv5_3'] = self.build_vgg_conv(vgg,self.model['conv5_2'],'conv5_3')
            self.model['conv5_4'] = self.build_vgg_conv(vgg,self.model['conv5_3'],'conv5_4')
            self.model['pool5'] = self.build_vgg_pool(vgg,self.model['conv5_4'],'pool5')

    def build_vgg_conv(self,vgg, prev, name):
        with tf.variable_scope(name):
            filt = tf.constant(vgg[name][0],name='filter')
            conv = tf.nn.conv2d(prev,filt,[1,1,1,1],padding='SAME')
            bias = tf.constant(vgg[name][1],name='bias')

            return tf.nn.relu(conv + bias)

    def build_vgg_pool(self,vgg,prev,name):
        return tf.nn.avg_pool(prev,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name=name)


    def set_style_loss(self):

        def gram_matrix(conv,N,M):
            matrix = tf.reshape(conv,[M,N])
            return tf.matmul(tf.transpose(matrix), matrix)

        layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        loss = 0.

        for l in layers:
            style_layer = self.model[l]
            shapes = style_layer.get_shape().as_list() # NOTE: not use .as_list() get error when reshape
            N = shapes[3]
            M = shapes[1]*shapes[2]

            G = gram_matrix(style_layer,N,M)
            A = gram_matrix(style_layer,N,M)

            loss += 1./(4.* M**2 * N**2)* tf.reduce_sum(tf.pow(G-A,2)) * 1./5.

        return loss
        

    def set_content_loss(self):
        content_layer = self.model['conv4_2']
        N = content_layer.shape[3]
        M = content_layer.shape[1]*content_layer.shape[2]

        #loss = 1./(2. * M * N) * tf.reduce_sum(tf.pow((content_layer - self.model['conv4_2']),2))
        loss = 1./(2.) * tf.reduce_sum(tf.pow((content_layer - self.model['conv4_2']),2))
        return loss
