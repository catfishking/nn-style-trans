# nn-style-trans
Neural style transfer, the implementation of [Image Style Transfer Using Convolutional Neural Networks] (http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

# Requirements
tensorflow

# Usage
>*Please download vgg19.npy from [tensorflow-vgg] (https://github.com/machrisaa/tensorflow-vgg)*

`$ python style_trans.py [-h] [-o OUTPUT] [-v VGG19] content_img style_img`  

### Results
<img src=https://github.com/catfishking/nn-style-trans/blob/master/image/ntu2.png?raw=true width=250px>
<img src=https://github.com/catfishking/nn-style-trans/blob/master/image/StarryNight.jpg?raw=true width=250px>
<img src=http://i.imgur.com/YwDI90D.jpg width=250px>

# Referennce
https://github.com/log0/neural-style-painting  
https://github.com/machrisaa/tensorflow-vgg
