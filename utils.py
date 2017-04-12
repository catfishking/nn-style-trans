import numpy as np
from PIL import Image

VGG_MEAN = [103.939, 116.779, 123.68]

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

