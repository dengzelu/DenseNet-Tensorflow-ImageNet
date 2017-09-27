import tensorflow as tf
import numpy as np


# ---------------------------------------------------------------------------------------------------------
# high-level network components start (dense block and transition layer)
def dense_block(input_, num_bl, name, growth_rate):
    with tf.variable_scope(name):
        output = input_
        for i in range(num_bl):
            output = bottleneck_layer(output, name='bottleneck_x%d' % (i + 1), growth_rate=growth_rate)
    return output


def transition_layer(input_, name, theta):
    with tf.variable_scope(name):
        input_channel = int(input_.get_shape()[-1])
        assert int(input_channel * theta) == input_channel * theta
        output_channel = int(input_channel * theta)

        output = batch_norm(input_, relu=True, name='bn')
        output = conv2d(output, output_channel=output_channel, ksize=1, stride=1, name='conv')
        output = avg_pool(output, ksize=2, stride=2, name='pool')
    return output
# high-level network components end
# ----------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# middle-level network components start
# bottleneck_layer (BN-Relu-Conv1x1-BN-Relu-Conv3x3)
def bottleneck_layer(input_, name, growth_rate):
    with tf.variable_scope(name):
        output = batch_norm(input_, relu=True, name='bn_x1')
        output = conv2d(output, output_channel=growth_rate * 4, ksize=1, stride=1, name='conv_x1')
        output = batch_norm(output, relu=True, name='bn_x2')
        output = conv2d(output, output_channel=growth_rate, ksize=3, stride=1, name='conv_x2')
        output = tf.concat((input_, output), axis=3)
    return output
# middle-level network components end
# -----------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# low-level network components start (conv2d max_pool avg_pool batch_norm and make_variable)
def conv2d(input_, output_channel, ksize, stride, name, padding='SAME'):
    with tf.variable_scope(name):
        input_channel = int(input_.get_shape()[-1])
        shape = [ksize, ksize, input_channel, output_channel]
        weights = make_variable(shape, 'weights')
        output = tf.nn.conv2d(input_, weights, strides=[1, stride, stride, 1], padding=padding)
    return output


def max_pool(input_, ksize, stride, name, padding='SAME'):
    return tf.nn.max_pool(
        input_, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding, name=name
    )


def avg_pool(input_, ksize, stride, name, padding='SAME'):
    return tf.nn.avg_pool(
        input_, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding, name=name
    )


def batch_norm(input_, relu, name):
    with tf.variable_scope(name):
        input_channel = int(input_.get_shape()[-1])
        mean = make_variable([1, 1, 1, input_channel], name='mean')
        variance = make_variable([1, 1, 1, input_channel], name='variance')
        offset = make_variable([1, 1, 1, input_channel], name='offset')
        scale = make_variable([1, 1, 1, input_channel], name='scale')
        variance_epsilon = 1e-5
        output = tf.nn.batch_normalization(
            input_, mean=mean, variance=variance, offset=offset, scale=scale, variance_epsilon=variance_epsilon
        )
        if relu:
            output = tf.nn.relu(output)
        return output


def make_variable(shape, name):
    var = tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0)
    )
    return var
# low-level network components end
# -------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# the following function helps load .npy weights for this model

    # load the weights file, the following introduce how to use npy weights_file
    # the weights_file is converted from caffemodel
    # use weights['conv4_32_x1']['weights'] or weights['conv4_32_x1_bn']['mean']to get the array
    # conv4 means the 3th dense block (because there is an 7×7 conv op named conv1)
    # 32 means 32th bottleneck layer(a bottleneck layer: BN-Relu-Conv(1×1)-BN-Relu-Conv(3×3) )
    # x1 means the first conv (1×1 conv) and x1_bn means the fist BN
    # x2 means the second conv (3×3 conv) and x2_bn means the second BN
    # for weights['conv4_32_x1'], there are only 'weights' key except for weights['fc6']
    # for weights['conv4_32_x1_bn'], there are 'mean', 'variance', 'offset', 'scale' keys

    # due to the different name strategy
    # we must map the name in tensorflow to the name in weights_file
    # such as 'block_x3/bottleneck_x32/conv_x1/weights:0'] to ['conv4_32_x1']['weights']
    # this is exactly what the following will do

def load_weights(weights_dir, sess):
    weights = np.load(weights_dir, encoding='latin1').item()
    map_variable = _get_map()
    variables = tf.trainable_variables()
    for var in variables:
        full_name = var.name.split('/')
        var_name = full_name[-1]
        var_name = var_name.split(':')[0]
        del full_name[-1]
        prefix_name = '/'.join(full_name)
        sess.run(var.assign(weights[map_variable[prefix_name]][var_name]))


def _get_map():
    # DenseNet_161 (4 dense blocks and [6, 12, 36, 24] bottleneck layers in each block)
    num_block = 4
    num_bl = [6, 12, 36, 24]
    map_variable = dict()
    # first conv and bn
    map_variable['block_x0/conv'] = 'conv1'
    map_variable['block_x0/bn'] = 'conv1_bn'

    # for all dense blocks
    for i in range(1, num_block + 1):
        for j in range(1, num_bl[i - 1] + 1):
            map_variable['block_x%d/bottleneck_x%d/bn_x1' % (i, j)] = 'conv%d_%d_x1_bn' % (i + 1, j)
            map_variable['block_x%d/bottleneck_x%d/conv_x1' % (i, j)] = 'conv%d_%d_x1' % (i + 1, j)
            map_variable['block_x%d/bottleneck_x%d/bn_x2' % (i, j)] = 'conv%d_%d_x2_bn' % (i + 1, j)
            map_variable['block_x%d/bottleneck_x%d/conv_x2' % (i, j)] = 'conv%d_%d_x2' % (i + 1, j)

    # for all transition layers
    for i in range(1, 4):
        map_variable['transition_x%d/bn' % i] = 'conv%d_blk_bn' % (i + 1)
        map_variable['transition_x%d/conv' % i] = 'conv%d_blk' % (i + 1)

    # for the final fully connected
    map_variable['classification/bn'] = 'conv5_blk_bn'
    map_variable['classification/fc'] = 'fc6'
    return map_variable

# function load_weights ends
# ---------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------
# do you need this function?
# for ImageNet data, there may be two different label strategy , one is the official labels which range [1, 1000],
# the other is the obtained by ascending rank of the WNID, which range [0, 999]
# the latter is much more popular than former, and it looks like in /Resources/ascending_order.txt (note the WNID !)
# so if you using latter label, you don't need this function
# this function helps to map the ascending WNID labels to official labels (not [1, 1000] but [0, 999] just minus 1)
# for those who (for example, me) are using the ImageNet official style label, must use this function

def map_labels(official_order_dir, ascending_order_dir):

    # the following get the official WNID order, for example ['n02119789', 'n02100735', 'n02110185', ...]
    with open(official_order_dir, 'r') as fr:
        official_order_temp = fr.readlines()
    assert len(official_order_temp) == 1000
    official_wnid_order = []
    for index in range(1000):
        official_wnid_order.append(official_order_temp[index].split(';')[0])

    # the following get the ascending WNID order, for example ['n01440764', 'n01443537', 'n01484850', ...]
    with open(ascending_order_dir, 'r') as fr:
        ascending_order_temp = fr.readlines()
    assert len(ascending_order_temp) == 1000
    ascending_wnid_order = []
    for index in range(1000):
        ascending_wnid_order.append(
            ascending_order_temp[index].split(' ')[0]
        )

    # the following find a mapping between two order
    maps = []
    for wnid in ascending_wnid_order:
        location = official_wnid_order.index(wnid)
        maps.append(location)
    assert len(maps) == 1000
    return maps


def map_logits(logits_array, maps):
    """
    Args:
        logits_array: [batch_size, 1000] numpy array, the predicted logits
        maps: a list, map[i] = j means i label in ascending WNID order is j label in official WNID order
    Returns:
        mapped_logits: [batch_size, 1000] numpy array, the predicted logits
    """

    assert logits_array.shape[1] == 1000, (
        (' number of probabilities does not equal 1000\n'
         'this message comes from function mapping_probs')
    )
    mapped_logits = np.zeros_like(logits_array)
    for index in range(1000):
        mapped_logits[:, maps[index]] = logits_array[:, index]

    return mapped_logits

# the end
# ---------------------------------------------------------------------------------------------------------------
