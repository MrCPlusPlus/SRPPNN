import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.000001)

def Conv2D(x, channel, kernel_size, stride=1, bias=False, padding='SAME',name="conv2d",reuse=False):
    with tf.variable_scope(name):
        initializer=tf.random_normal_initializer(stddev=0.001)
        conv_out = tf.layers.conv2d(inputs=x,use_bias=bias,kernel_initializer=initializer,kernel_regularizer=weight_regularizer,filters=channel,kernel_size=kernel_size,strides=stride,padding=padding,reuse=reuse)
        return conv_out

def Residual_Block(name, x, num_channels):
    with tf.variable_scope(name):
        skip = x
        x = Conv2D(x,num_channels,3,padding='same',name='conv0')
        x = tf.nn.relu(x)
        x = Conv2D(x,num_channels,3,padding='same',name='conv1')
        return x + skip 
    
def Concat(layers):
    return tf.concat(layers, axis=3)

def ReLU(x):
    return tf.nn.relu(x)

def LRelu(x):
    return tf.maximum(x*0.2,x)

def gaussian_kernel(size=5,sigma=1):
    x_points = np.arange(-(size-1)//2,(size-1)//2+1,1)
    y_points = x_points[::-1]
    xs,ys = np.meshgrid(x_points,y_points)
    kernel = np.exp(-(xs**2+ys**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernel/kernel.sum()

class SRPPNN():
  def __init__(self, scale=4):
    if scale==2:
        scale = 3
    self.scale = scale
  
  def _srppnn(self,inputs_mul,inputs_pan,reuse=False):
      with tf.variable_scope("SRPPNN",reuse=reuse) as vs:
        middle_size_h = inputs_mul.shape[1]*self.scale//2
        middle_size_w = inputs_mul.shape[2]*self.scale//2
        output_size_h = inputs_pan.shape[1]
        output_size_w = inputs_pan.shape[2]
        inputs_mul_up_p1 = tf.image.resize_bicubic(inputs_mul, [middle_size_h, middle_size_w])
        inputs_mul_up_p2 = tf.image.resize_bicubic(inputs_mul, [output_size_h, output_size_w])
        kernel = gaussian_kernel(11, 1)
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        inputs_pan_blur = tf.nn.conv2d(inputs_pan, kernel, strides=[1, 1, 1, 1], padding='SAME')
        inputs_pan_down_p1 = tf.image.resize_bicubic(inputs_pan_blur, [middle_size_h, middle_size_w])

        pre_inputs_mul_p1_feature = Conv2D(inputs_mul, 32, [3,3], 1, True,name='conv_mul_pre_p1')
        x = pre_inputs_mul_p1_feature
        for d in range(4):
          x = Residual_Block('img_mul_p1_layer{}'.format(d),x,32)
        post_inputs_mul_p1_feature = Conv2D(x, 32, [3,3], 1, True,name='conv_mul_post_p1')
        inputs_mul_p1_feature = pre_inputs_mul_p1_feature+post_inputs_mul_p1_feature
        inputs_mul_p1_feature_bic = tf.image.resize_bicubic(inputs_mul_p1_feature, [middle_size_h, middle_size_w], True)
        net_img_p1_sr = Conv2D(inputs_mul_p1_feature_bic, inputs_mul.shape[3], [3,3], 1, True,name='mul_grad_p1') + inputs_mul_up_p1
        inputs_p1 = Concat([net_img_p1_sr,inputs_pan_down_p1])

        pre_inputs_p1_feature = Conv2D(inputs_p1, 32, [3,3], 1, True,name='conv_pre_p1')
        x = pre_inputs_p1_feature
        for d in range(4):
          x = Residual_Block('img_p1_layer{}'.format(d),x,32)
        post_inputs_p1_feature = Conv2D(x, 32, [3,3], 1, True,name='conv_post_1')
        inputs_p1_feature = pre_inputs_p1_feature+post_inputs_p1_feature

        inputs_pan_down_p1_blur = tf.nn.conv2d(inputs_pan_down_p1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        inputs_pan_hp_p1 = inputs_pan_down_p1 - inputs_pan_down_p1_blur
        net_img_p1 = Conv2D(inputs_p1_feature, inputs_mul.shape[3], [3,3], 1, True,name='grad_p1') + inputs_mul_up_p1 + inputs_pan_hp_p1

        pre_inputs_mul_p2_feature = Conv2D(net_img_p1, 32, [3,3], 1, True,name='conv_mul_pre_p2')
        x = pre_inputs_mul_p2_feature
        for d in range(4):
          x = Residual_Block('img_mul_p2_layer{}'.format(d),x,32)
        post_inputs_mul_p2_feature = Conv2D(x, 32, [3,3], 1, True,name='conv_mul_post_p2')
        inputs_mul_p2_feature = pre_inputs_mul_p2_feature+post_inputs_mul_p2_feature
        inputs_mul_p2_feature_bic = tf.image.resize_bicubic(inputs_mul_p2_feature, [output_size_h, output_size_w], True)
        net_img_p2_sr = Conv2D(inputs_mul_p2_feature_bic, inputs_mul.shape[3], [3,3], 1, True,name='mul_grad_p2') + inputs_mul_up_p2
        inputs_p2 = Concat([net_img_p2_sr,inputs_pan])
        
        pre_inputs_p2_feature = Conv2D(inputs_p2, 32, [3,3], 1, True,name='conv_pre_p2')
        x = pre_inputs_p2_feature
        for d in range(4):
          x = Residual_Block('img_p2_layer{}'.format(d),x,32)
        post_inputs_p2_feature = Conv2D(x, 32, [3,3], 1, True,name='conv_post_p2')
        inputs_p2_feature = pre_inputs_p2_feature+post_inputs_p2_feature

        inputs_pan_hp_p2 = inputs_pan - inputs_pan_blur
        net_img_p2 = Conv2D(inputs_p2_feature, inputs_mul.shape[3], [3,3], 1, True,name='grad_p2') + inputs_mul_up_p2 + inputs_pan_hp_p2

      return net_img_p2

def L2_loss(outputs, labels):
    loss = tf.reduce_mean((tf.square(outputs-labels)), name='L2_loss')
    return [loss]

def training(loss, lr):
    optimizer = tf.train.AdamOptimizer(lr)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

