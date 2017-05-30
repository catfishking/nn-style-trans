import numpy as np
import tensorflow as tf
import pandas as pd
import random

from os import listdir
from os.path import isfile, join, splitext, basename
from PIL import Image

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True


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
    if array.dtype is not np.dtype('int8') :
        array = np.clip(array,0,255).astype('int8')
    im = Image.fromarray(array,'RGB')
    im.save(out_path)

def image_generator(filenames, batch_size=30):
    data_batch = []
    count = 0
    while True:
        random.shuffle(filenames) # shuffle file list
        for f in filenames:
            count += 1
            data = load_rgb(f)
            data_batch.append(data)

            if count == batch_size:
                data_batch = np.array(data_batch)
                data_batch = data_batch.astype(np.float32)
                data_batch = np.squeeze(data_batch, axis=1) # remove reduntant dim
                yield data_batch
                count = 0
                data_batch = []
            

def coco_input(path, batch_size = 30, img_size=256):
    img_files = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
    filename_q = tf.train.string_input_producer(img_files,shuffle=True)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_q)
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.int8)
    image = tf.reshape(image, [img_size,img_size,3])

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


def build_wikiArt_TFRecord(path, out_dir):
    # read wikiart.data
    wikiart_labels = pd.read_csv(join(path,'wikiart.data'), skiprows=10)
    wikiart_labels = wikiart_labels[['contentId', 'artistUrl']].values
    image_path = join(path, 'images_512')
    filenames = [join(image_path,f) for f in listdir(image_path) if isfile(join(image_path, f))]
    contentID_artist_map = {}
    artist_ID_map = {}
    for i in wikiart_labels:
        contentID_artist_map[i[0]] = i[1]
        if i[1] not in artist_ID_map:
            artist_ID_map[i[1]] = len(artist_ID_map)

    for style_file in filenames:
        image = load_rgb(style_file)
        style_file_noext = splitext(basename(style_file))[0]

        label = contentID_artist_map[int(style_file_noext)]
        label = artist_ID_map[label]

        # Convert to binary
        label = np.int32(label).tobytes()
        image = image.tobytes()
        tfrecord_filename = join(out_dir, style_file_noext) + '.tfrecord'
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'label':tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[label])),
                        'data':tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image]))
                        }
                    )
                )
        writer.write(example.SerializeToString())
        writer.close()


def read_wikiArt_TFRecord(TFRecord_dir, batch_size=10):
    filenames = [join(TFRecord_dir,f) for f in listdir(TFRecord_dir) if isfile(join(TFRecord_dir, f))]
    tfrecordfile_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord = reader.read(tfrecordfile_queue)

    tfrecord_features = tf.parse_single_example(tfrecord,
                        features={
                            'label':tf.FixedLenFeature([], tf.string),
                            'data':tf.FixedLenFeature([], tf.string),
                            }, name = 'features')
    data = tf.decode_raw(tfrecord_features['data'], tf.float32)
    data = tf.reshape(data,[512,512,3]) # NOTE  shuffle_batch require specific shape
    #data = tf.reshape(data,shape)

    label = tf.decode_raw(tfrecord_features['label'], tf.int32)
    label = tf.one_hot(label,depth=4)
    label = tf.reshape(label,[4]) # NOTE shuffle_batch_require specific shape

    min_after_dequeue = 10 * batch_size
    capacity = 20 * batch_size

    data_batch, label_batch = tf.train.shuffle_batch([data, label], batch_size=batch_size, 
            capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch
    #return data, label


def read_batches(dir_path):
    data_batch, label_batch = read_wikiArt_TFRecord(dir_path)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(10): # generate 10 batches
            features, labels = sess.run([data_batch, label_batch])
            print (features.shape, labels.shape)
            print (labels)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    '''
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
    '''
    #images = read_COCO('/tmp3/troutman/COCO/train2014_256')
    #print images.shape
    out_dir = '/tmp3/troutman/WikiArt/wikiart/ha/TFRecord/'
    #build_wikiArt_TFRecord('/tmp3/troutman/WikiArt/wikiart/ha', out_dir)

    read_batches(out_dir)
