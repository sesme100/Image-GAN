    
import cv2

import numpy as np
import tensorflow as tf

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def Generator(x, isTraining, reuse = False, name = 'Generator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        if not reuse:
            print('Generator input :', x)
            
        x = tf.layers.dense(x, units = 128)
        x = lrelu(x, 0.2)

        x = tf.layers.dense(x, units = 256)
        x = lrelu(x, 0.2)
        
        x = tf.layers.dense(x, units = 784)
        x = tf.nn.tanh(x)

        if not reuse:
            print('Generator output :', x)

        return x

def Discriminator(x, isTraining, reuse = False, name = 'Discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        if not reuse:
            print('Discriminator input :', x)
            
        x = tf.layers.dense(x, units = 256, name = 'fc1')
        x = lrelu(x, 0.2)

        x = tf.layers.dense(x, units = 128, name = 'fc2')
        x = lrelu(x, 0.2)

        _x = tf.layers.dense(x, units = 1, name = 'fc3')
        x = tf.nn.sigmoid(_x)

        if not reuse:
            print('Discriminator output :', x)

        return x, _x
