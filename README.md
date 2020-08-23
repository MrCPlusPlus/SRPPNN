# SRPPNN
The model code of "Super-Resolution-Guided Progressive Pansharpening based on a Deep Convolutional Neural Network"  
[[paper]](https://ieeexplore.ieee.org/document/9172104)

This repo includes the main code of the SRPPNN (with Tensorflow). To use this model:

1. import this .py file into your file that wants to create SRPPNN  

```from SRPPNN_model import *```

2. Create the SRPPNN model by  

```
input_mul_pl = tf.placeholder(tf.float32, shape = (None, img_rows//scale, img_cols//scale, num_bands), name='input_mul') //Low-resolution multispectral images. Scale is the parameter that represents ratio between panchromatic and multispectral images.
input_pan_pl = tf.placeholder(tf.float32, shape = (None, img_rows, img_cols, 1), name='input_pan') //High-resolution panchromatic images
lbl_pl = tf.placeholder(tf.float32, shape = (None, img_rows, img_cols, num_bands), name='label') //Original multispectral images
net = SRPPNN(scale=scale)  //the default scale is 4
outputs = net._srppnn(input_mul_pl,input_pan_pl) 
```
   
3. loss function and optimizer  
```
loss = L2_loss(outputs, lbl_pl)
train_op = training(loss[0], l_rate) //l_rate: learning rate
```

4. Train the model like the typical Tensorflow model
