from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset
import tensorflow as tf
BATCH_SIZE = 32
CLASS_NUM = 10
# FILTER_SIZES = [64, 128, 256, 512]
FILTER_SIZES = [32, 64, 128, 256]
MODULE_SIZES = [2, 2, 2, 2]
WEIGHT_DECAY = 1e-4
def preactivation_block(input, num_filters, stride=1):
    num_filters_in = input.get_shape().as_list()[1]

    # residual
    net = BNReLU(input)
    residual = Conv2D('conv1', net, num_filters, kernel_size=3, strides=stride, use_bias=False, activation=BNReLU)
    residual = Conv2D('conv2', residual, num_filters, kernel_size=3, strides=1, use_bias=False)

    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = Conv2D('shortcut', net, num_filters, kernel_size=1, strides=stride, use_bias=False)

    return shortcut + residual


class ResNet_Cifar(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 96, 96, 1], 'input'),
                tf.placeholder(tf.float32, [None, 10], 'label')]

    def build_graph(self, image, label):
        assert tf.test.is_gpu_available()

        MEAN_IMAGE = tf.constant([0.4822], dtype=tf.float32)#tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        STD_IMAGE = tf.constant([0.1994], dtype=tf.float32) #tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE
        image = tf.transpose(image, [0, 3, 1, 2])

        pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
        with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, kernel_initializer=pytorch_default_init):
            net = Conv2D('conv0', image, 64, kernel_size=3, strides=1, use_bias=False)
            for i, blocks_in_module in enumerate(MODULE_SIZES):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = preactivation_block(net, FILTER_SIZES[i], stride)
            net = GlobalAvgPooling('gap', net)
            logits = FullyConnected('linear', net, CLASS_NUM,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

        ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
        ce_cost = tf.reduce_mean(ce_cost, name='cross_entropy_loss')

        single_label = tf.to_int32(tf.argmax(label, axis=1))
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), ce_cost)
        add_param_summary(('.*/W', ['histogram']))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')

        return tf.add_n([ce_cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

