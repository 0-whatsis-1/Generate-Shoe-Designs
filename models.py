import os
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import scipy.misc
from process import*

HEIGHT, WIDTH = 64, 64
CHANNEL = 3
BATCH_SIZE = 64
EPOCH = 1000
VERSION = 'newDesigns'
newShoe_path = './' + VERSION


def generator(input, rand_dim, is_train, reuse=False):
    """
    # input = length of latent vector(i.e. 100)
    # Depth of feature maps carried through the generator is 64
    # Number of channels in the output image is 3
    # Creates a RGB image with the same size as the training image(64, 64)
    # First input is reshaped before feeding to the model
    # Consists of a series of strided 2D Convolutional Transpose layers paired
      with a 2D Batch Normalization and a ReLU activation and output is fed
      through a tanh Activation function to return it to the input data range [-1, 1].
    """
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        # custom weight & bias initialization
        w1 = tf.get_variable('w1', shape=[rand_dim, 4 * 4 * 512], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1', shape=[512 * 4 * 4], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        Dense = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # Reshape -> Batch Normalization -> Activation(ReLU)
        conv1 = tf.reshape(Dense, shape=[-1, 4, 4, 512], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')  # OUTPUT is: 4*4*512
        # Strided Convolutional Transpose -> Batch Normalization -> Activation(ReLU)
        conv2 = tf.layers.conv2d_transpose(act1, 256, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')  # OUTPUT is: 8*8*256

        conv3 = tf.layers.conv2d_transpose(act2, 128, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')  # OUTPUT is: 16*16*128

        conv4 = tf.layers.conv2d_transpose(act3, 64, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')  # OUTPUT is: 32*32*64
        # Strided Convolutional Transpose -> Activation(tanh)
        conv5 = tf.layers.conv2d_transpose(act4, 3, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5')
        out = tf.nn.tanh(conv5, name='out')  # OUTPUT is: 64*64*3

        return out


def discriminator(input, is_train, reuse=False):
    """
    # Depth of feature maps propogated through the discriminator is 64
    # It takes an input imag of size 64*64*3
    # Consists of a series of strided 2D Convolutional layers paired with
      a 2D Batch Normalization and a Leaky ReLU activation and outputs the
      final probability through the use of Linear Activation.
    """
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        # Strided Convolution -> Batch Normalization -> Activation(Leaky-ReLU)
        conv1 = tf.layers.conv2d(input, 64, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = tf.nn.leaky_relu(conv1, alpha=0.2, name='act1')  # OUTPUT is: 32*32*64

        conv2 = tf.layers.conv2d(act1, 128, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.leaky_relu(bn2, alpha=0.2, name='act2')  # OUTPUT is: 16*16*128

        conv3 = tf.layers.conv2d(act2, 256, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.leaky_relu(bn3, alpha=0.2, name='act3')  # OUTPUT is: 8*8*256

        conv4 = tf.layers.conv2d(act3, 512, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.leaky_relu(bn4, alpha=0.2, name='act4')  # OUTPUT is: 4*4*512
        # using Linear Activation {W-GAN}
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(fc1, w2), b2, name='out')

        return out


def train():
    rand_dim = 100
    total_batch = 0
    disc_iterations, genr_iterations = 5, 1
    disc_loss, genr_loss = 0, 0

    with tf.variable_scope('input'):
        real_shoe = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_shoe')
        random_input = tf.placeholder(tf.float32, shape=[None, rand_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    fake_shoe = generator(random_input, rand_dim, is_train)
    real_test = discriminator(real_shoe, is_train)
    fake_test = discriminator(fake_shoe, is_train, reuse=True)
    # Use of Wasserstein loss to train that promotes larger difference between scores for real & fake images
    loss_of_disc = tf.reduce_mean(fake_test) - tf.reduce_mean(real_test)
    loss_of_genr = -tf.reduce_mean(fake_test)

    vars_to_train = tf.trainable_variables()
    disc_var = [var for var in vars_to_train if 'dis' in var.name]
    genr_var = [var for var in vars_to_train if 'gen' in var.name]
    trainer_disc = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_of_disc, var_list=disc_var)
    trainer_genr = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_of_genr, var_list=genr_var)
    # Constrain Discriminator weights to a limited range after each mini batch{W-GAN}
    clip_disc = [val.assign(tf.clip_by_value(val, -0.01, 0.01)) for val in disc_var]

    img_batch, samples_num = data_preprocess()
    batch_num = int(samples_num / BATCH_SIZE)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    save_path = saver.save(sess, "./modelEdited.checkpoint")
    checkpoint = tf.train.latest_checkpoint('./modelEdited/' + VERSION)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Total number of training samples:%d' % samples_num)
    print('Batch size: %d, Batch number per epoch: %d, Epoch number: %d' % (BATCH_SIZE, batch_num, EPOCH))
    print('Training starts -----')
    for i in range(EPOCH):
        print("Running epoch {}/{}...".format(i, EPOCH))
        for j in range(batch_num):
            print('Batch number ---', j)
            train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, rand_dim]).astype(np.float32)
            for k in range(disc_iterations):
                print('Updating Discriminator', k)
                train_image = sess.run(img_batch)
                sess.run(clip_disc)
                _, disc_loss = sess.run([trainer_disc, loss_of_disc],
                    feed_dict={random_input: train_noise, real_shoe: train_image, is_train: True})
            for k in range(genr_iterations):
                print('Updating Generator', k)
                _, genr_loss = sess.run([trainer_genr, loss_of_genr],
                    feed_dict={random_input: train_noise, is_train: True})

        if(i % 200 == 0):
            if not os.path.exists('./modelEdited/' + VERSION):
                os.makedirs('./modelEdited/' + VERSION)
            saver.save(sess, './modelEdited/' +VERSION + '/' + str(i))
        if(i % 100 == 0):
            if not os.path.exists(newShoe_path):
                os.makedirs(newShoe_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, rand_dim]).astype(np.float32)
            imgtest = sess.run(fake_shoe, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8, 8] , newShoe_path + '/epoch' + str(i) + '.jpg')
            print('Training:[%d], Loss of Discriminator:%f, Loss of Generator:%f' % (i, disc_loss, genr_loss))

    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
    train()
