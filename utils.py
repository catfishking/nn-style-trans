import numpy as np
import tensorflow as tf

from os import listdir
from os.path import isfile, join
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth=True


def load_rgb(img_path):
    ''' Load RGB image then return a 4d(1,R,G,B) numpy array'''
    im = Image.open(img_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image = np.array(list(im.getdata()),dtype='float32')
    #image = image.astype('float32')
    image = image.reshape(1,im.size[1],im.size[0],3)
    return image

def save_rgb(out_path,array):
    ''' Save RGB image'''
    array = array.reshape(array.shape[1:])
    array = array
    array = np.clip(array,0,255).astype('int8')
    im = Image.fromarray(array,'RGB')
    im.save(out_path)


def coco_input(path, batch_size = 30):
    img_files = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
    filename_q = tf.train.string_input_producer(img_files,shuffle=True)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_q)
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.int8)
    image = tf.reshape(image, [256,256,3])

    #bbox_begin = (0,0,0)
    #bbox_size = tf.constant((256, 256, 3), dtype=tf.int32)
    #crop_image = tf.slice(image, bbox_begin, bbox_size)

    min_after_dequeue = 10 * batch_size
    capacity = 30 * batch_size
    
    data_batch = tf.train.shuffle_batch([image], batch_size=batch_size, 
            capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch

def read_COCO(path):
    # XXX don't use it
    ''' Return a huge 4-d array of images'''
    images = []
    img_files = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
    for img in img_files:
        images.append(load_rgb(img))
    images = np.array(images)
    return images

if __name__ == '__main__':
    
    data_batch = coco_input('/tmp3/troutman/COCO/train2014_256', 1)
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for _ in range(100000):
            a = sess.run(data_batch)
            print a.shape
            save_rgb('test.png',a)
        coord.request_stop()
        coord.join(threads)
    
    #images = read_COCO('/tmp3/troutman/COCO/train2014_256')
    #print images.shape
    





