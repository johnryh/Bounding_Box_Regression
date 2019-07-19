from config import *
import tensorflow as tf
import numpy as np
import os, math

def activation_func(z, activation='leaky_relu'):
    if activation == 'leaky_relu':
        return tf.nn.leaky_relu(z)
    elif activation == 'relu':
        return tf.nn.relu(z)
    elif activation == 'selu':
        return tf.nn.selu(z)
    elif activation == 'tanh':
        return tf.nn.tanh(z)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(z)
    elif activation == 'linear':
        return z

    assert False, 'Activation Func "{}" not Found'.format(activation)


def PN(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[3], s[1], s[2]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.

        return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.


def dense(z, units, activation=None, name='Dense', gain=np.sqrt(2)/4, use_PN=False):
    with tf.variable_scope(name):
        with tf.device("/device:{}:0".format(controller)):
            assert len(z.shape) == 2, 'Input Dimension must be rank 2, but is rank {}'.format(len(z.shape))
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=gain)
            weights = get_weight([z.shape[1].value, units], gain, use_wscale=True)
            biases = tf.get_variable('bias', [units], initializer=initializer)

            y = tf.add(tf.matmul(z, weights), biases)

            if activation:
                y= activation_func(y, activation)

            if use_PN:
                y = PN(y)

            return y


def conv2d(input_vol, num_kernal, scope, kernal_size=3, stride=1, activation='leaky_relu', padding='SAME', batch_norm=False, gain=np.sqrt(2), use_PN=False):
    with tf.variable_scope(scope):
        if isinstance(kernal_size, int):
            kernal_height = kernal_size
            kernal_width = kernal_size
        else:
            kernal_height = kernal_size[0]
            kernal_width = kernal_size[1]

        with tf.device("/device:{}:0".format(controller)):
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=gain)
            weights = get_weight([kernal_height, kernal_width, int(input_vol.shape[-1]), int(num_kernal)], gain, use_wscale=True)
            biases = tf.get_variable('bias', [int(num_kernal)], initializer=initializer)

        conv = tf.add(tf.nn.conv2d(input_vol, weights, [1, stride, stride, 1], padding=padding), biases)

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=True)

        out = activation_func(conv, activation)

        if use_PN:
            out = PN(out)

        return out


def IoU(logits, label):
    pass


class BB_Regressor():

    def __init__(self, img, label):
        self.img = img
        self.label = label
        self.initialize_optimizer()
        self.build_model(img, label)
        self.model_stats()

    def initialize_optimizer(self):
        with tf.variable_scope('GAN_Optimizer'):
            with tf.variable_scope('Optim'):
                self.optim = tf.train.AdamOptimizer(learning_rate=ln_rate, beta1=0.0, beta2=0.99, epsilon=1e-8, name='d_optim') # was 0.5, 0.9
        print('Solver Configured')

    def build_model(self, img, label):
        img_split = tf.split(img, num_gpus, name='img_split') if num_gpus > 1 else [img]
        label_split = tf.split(label, num_gpus, name='label_split') if num_gpus > 1 else [label]

        tower_score=[]; tower_loss=[]; tower_grads = [];
        for gpu_id in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope('Regressor') as GAN_scope:
                    blocks = [img_split[gpu_id]]
                    #num_features = [16, 32, 64, 128]
                    #num_features = [8, 8, 8, 8]
                    num_features = [8, 16, 32, 64]

                    for i in range(4):
                        blocks.append(conv2d(blocks[-1], num_features[i], 'conv{}'.format(len(blocks)), kernal_size=3, stride=1, activation='leaky_relu'))
                        blocks.append(conv2d(blocks[-1], num_features[i], 'conv{}'.format(len(blocks)), kernal_size=3, stride=2, activation='leaky_relu'))

                    logits = dense(tf.reshape(blocks[-1], [int(batch_size/num_gpus), 7*7*num_features[i]]), 4, activation='sigmoid')

                    print(logits, label_split[gpu_id])
                    loss = tf.reduce_mean(tf.square((logits - label_split[gpu_id])))
                    tower_loss.append(loss)
                    tower_score.append(logits)

                with tf.variable_scope('Compute_Optim_Gradients'):
                    tower_grads.append(self.optim.compute_gradients(loss))

        self.saver = tf.train.Saver(name='Saver', max_to_keep=None)

        with tf.variable_scope('Sync_Point'):
            self.loss = tf.reduce_mean(tower_loss, axis=0, name='loss')
            self.logits = tf.concat(tower_score, axis=0)
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('GAN_Solver'):
            with tf.variable_scope('Apply_Optim_Gradients'), tf.device("/device:{}:1".format(controller)):
                self.grads = self.average_gradients(tower_grads)
                self.apply_grad = self.optim.apply_gradients(self.grads, name='Apply_Grads')


    def average_gradients(self, tower_grads):
        with tf.variable_scope('Average_Gradients'):

            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = [g for g, _ in grad_and_vars]
                grad = tf.reduce_mean(grads, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)

                average_grads.append(grad_and_var)
            return average_grads


    def model_stats(self):
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GAN/Discriminator'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Discriminator Total parameters:{}M'.format(total_parameters / 1e6))

        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GAN/Generator'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Generator Total parameters:{}M'.format(total_parameters / 1e6))

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total Total parameters:{}M'.format(total_parameters / 1e6))