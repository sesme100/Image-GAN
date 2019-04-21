import cv2
import time

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from GAN import *
from utils import *
from Define import *

#db load
mnist = input_data.read_data_sets("../../DB/MNIST/", one_hot = True)

#placeholder
x = tf.placeholder(tf.float32, shape = [None, 784])
z = tf.placeholder(tf.float32, shape = [None, 100])
isTraining = tf.placeholder(dtype = tf.bool)

#model
G_z = Generator(z, isTraining, reuse = False)

D_real, D_real_logits = Discriminator(x, isTraining, reuse = False)
D_fake, D_fake_logits = Discriminator(G_z, isTraining, reuse = True)

#loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logits, labels = tf.ones([BATCH_SIZE, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.zeros([BATCH_SIZE, 1])))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.ones([BATCH_SIZE, 1])))

#select variables
vars = tf.trainable_variables()
D_vars = [var for var in vars if var.name.startswith('Discriminator')]
G_vars = [var for var in vars if var.name.startswith('Generator')]

#optimizer
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_train = tf.train.AdamOptimizer(LEARNING_RATE, beta1 = 0.5).minimize(D_loss, var_list = D_vars)
    G_train = tf.train.AdamOptimizer(LEARNING_RATE, beta1 = 0.5).minimize(G_loss, var_list = G_vars)

#train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_images = (mnist.train.images - 0.5) / 0.5
    
    #fixed vector
    fixed_z = np.random.normal(0, 1, (100, 100))
    
    print('training...')
    for epoch in range(MAX_EPOCH):

        st_time = time.time()

        G_losses = []
        D_losses = []

        for iter in range(len(train_images) // BATCH_SIZE):
            batch_x = train_images[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]
            batch_z = np.random.normal(0, 1, (BATCH_SIZE, 100))

            _D_loss, _ = sess.run([D_loss, D_train], {x:batch_x, z:batch_z, isTraining:True})
            D_losses.append(_D_loss)

            sample_z = np.random.normal(0, 1, (BATCH_SIZE, 100))
            
            _G_loss, _ = sess.run([G_loss, G_train], {z:sample_z, x:batch_x, isTraining : True})
            G_losses.append(_G_loss)

        epoch_time = time.time() - st_time
        print('[%d/%d] - time : %.4fsec, D_loss : %.4f, G_loss : %.4f'%(epoch + 1, MAX_EPOCH, epoch_time, np.mean(D_losses), np.mean(G_losses)))

        data = sess.run(G_z, feed_dict={z:fixed_z, isTraining : False})

        imgs = np.zeros((SHOW_IMG_COUNT, 28, 28), dtype = np.uint8)
        for img_index in range(SHOW_IMG_COUNT):
            imgs[img_index] = ((data[img_index].reshape((28, 28)) + 1) * 127.5).astype(np.uint8)
        
        Save_Images(imgs, './results/{}.jpg'.format(epoch))

