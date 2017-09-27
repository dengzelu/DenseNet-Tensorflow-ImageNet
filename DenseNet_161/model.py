import tensorflow as tf

import utils


def inference(images, growth_rate=48, theta=0.5):
    with tf.variable_scope('pre_process'):
        r, g, b = tf.split(images, [1, 1, 1], axis=3)
        bgr_images = tf.concat([b, g, r], axis=3)
        mean_value = tf.constant([103.94, 116.78, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        scale = tf.constant(0.017, dtype=tf.float32, name='scale')
        value = (bgr_images - mean_value) * scale

    with tf.variable_scope('block_x0'):
        output = utils.conv2d(value, output_channel=96, ksize=7, stride=2, name='conv')
        output = utils.batch_norm(output, relu=True, name='bn')
        output = utils.max_pool(output, ksize=3, stride=2, name='pool')

    output = utils.dense_block(output, num_bl=6, name='block_x1', growth_rate=growth_rate)
    output = utils.transition_layer(output, name='transition_x1', theta=theta)
    output = utils.dense_block(output, num_bl=12, name='block_x2', growth_rate=growth_rate)
    output = utils.transition_layer(output, name='transition_x2', theta=theta)
    output = utils.dense_block(output, num_bl=36, name='block_x3', growth_rate=growth_rate)
    output = utils.transition_layer(output, name='transition_x3', theta=theta)
    output = utils.dense_block(output, num_bl=24, name='block_x4', growth_rate=growth_rate)

    with tf.variable_scope("classification"):
        output = utils.batch_norm(output, relu=True, name='bn')
        output = utils.avg_pool(output, ksize=7, stride=1, padding='VALID', name='pool')
        output = utils.conv2d(output, output_channel=1000, ksize=1, stride=1, name='fc')
        biases = utils.make_variable([1000], name='fc/biases')
        logits = tf.nn.bias_add(output, biases)
        logits = tf.squeeze(logits, axis=[1, 2])
    return logits
    
