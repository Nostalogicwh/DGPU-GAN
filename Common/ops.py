import tensorflow as tf
import numpy as np
import os
import sys
from tf_ops.grouping.tf_grouping import knn_point_2
from Common.model_utils import gen_1d_grid,gen_grid
sys.path.append(os.path.dirname(os.getcwd()))


# D 全连接输出置信值
def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        # out 256
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i
        )
    # out 1
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1)
    )
    return outputs


# D 全连接层提取特征
def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i
        )
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1)
    )
    return outputs


###
# 自注意力相关
###

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def instance_norm(net, train=True, weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable(
        'shift', shape=var_shape,
        initializer=tf.zeros_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
    )
    scale = tf.get_variable(
        'scale', shape=var_shape,
        initializer=tf.ones_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
    )
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv2d(
        inputs, num_output_channels, kernel_size,
        scope=None, stride=[1, 1], padding='SAME',
        use_xavier=True, stddev=1e-3, weight_decay=0.00001,
        activation_fn=tf.nn.relu,
        bn=False, ibn=False, bn_decay=None,
        use_bias=True, is_training=None,
        reuse=tf.AUTO_REUSE
        ):

    with tf.variable_scope(scope, reuse=reuse) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True
            )
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


# D 自注意力
def attention_unit(inputs, scope='attention_unit', is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(
            inputs, layer, [1, 1],
            padding='VALID',
            stride=[1, 1],
            bn=False,
            is_training=is_training,
            scope='conv_f',
            bn_decay=None
        )
        g = conv2d(
            inputs, layer, [1, 1],
            padding='VALID',
            stride=[1, 1],
            bn=False,
            is_training=is_training,
            scope='conv_g',
            bn_decay=None
        )
        h = conv2d(
            inputs, dim, [1, 1],
            padding='VALID',
            stride=[1, 1],
            bn=False,
            is_training=is_training,
            scope='conv_h',
            bn_decay=None
        )
        # 矩阵乘 [bs, N, N]
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
        # attention map
        beta = tf.nn.softmax(s, axis=-1)
        # [bs, N, N]*[bs, N, c] > [bs, N, c]
        o = tf.matmul(beta, hw_flatten(h))
        gamma = tf.get_variable('gamma', [1], initializer=tf.constant_initializer(0.0))
        # [bs, h, w, c]
        o = tf.reshape(o, shape=inputs.shape)
        x = gamma * o + inputs
    return x


# DGCNN patch-based特征提取模块
def conv1d(inputs, num_output_channels, kernel_size,
           scope=None, stride=1, padding='SAME',
           use_xavier=True, stddev=1e-3, weight_decay=0.00001,
           activation_fn=tf.nn.relu, bn=False, ibn=False,
           bn_decay=None, use_bias=True, is_training=None,
           reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx


def dense_conv(feature, n=3, growth_rate=64, k=16, scope='dense_conv', **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx


def dense_conv1(feature, n=3, growth_rate=64, k=16, scope='dense_conv', **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx


# G 特征提取
def feature_extraction(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                             padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                         bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                             padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                             padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                             padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features


# G 上采样模块
def up_block(inputs, up_ratio, scope='up_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim * up_ratio
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
        # grid = tf.expand_dims(grid, axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        net = attention_unit(net, is_training=is_training)

        net = conv2d(
            net, 256, [1, 1],
            padding='VALID', stride=[1, 1],
            bn=False, is_training=is_training,
            scope='conv1', bn_decay=bn_decay
        )
        net = conv2d(
            net, 128, [1, 1],
            padding='VALID', stride=[1, 1],
            bn=False, is_training=is_training,
            scope='conv2', bn_decay=bn_decay
        )
    return net


def down_block(inputs, up_ratio, scope='down_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = inputs
        net = tf.reshape(net, [tf.shape(net)[0], up_ratio, -1, tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(
            net, 256, [1, up_ratio],
            padding='VALID', stride=[1, 1],
            bn=False, is_training=is_training,
            scope='conv1', bn_decay=bn_decay
        )
        net = conv2d(
            net, 128, [1, 1],
            padding='VALID', stride=[1, 1],
            bn=False, is_training=is_training,
            scope='conv2', bn_decay=bn_decay
        )
    return net


def up_projection_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = conv2d(
            inputs, 128, [1, 1],
            padding='VALID', stride=[1, 1],
            bn=False, is_training=is_training,
            scope='conv0', bn_decay=bn_decay
        )

        H0 = up_block(L, up_ratio, is_training=is_training, bn_decay=bn_decay, scope='up_0')

        L0 = down_block(H0, up_ratio, is_training=is_training, bn_decay=bn_decay, scope='down_0')
        E0 = L0-L
        H1 = up_block(E0, up_ratio, is_training=is_training, bn_decay=bn_decay, scope='up_1')
        H2 = H0+H1
    return H2


def add_scalar_summary(name, value, collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])


def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])