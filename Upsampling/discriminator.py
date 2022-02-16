import tensorflow as tf
from Common import ops


class Discriminator(object):
    def __init__(self, opts, is_training, name='Discriminator'):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False
        self.start_number = 32

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            inputs = tf.expand_dims(inputs, axis=2)
            with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
                growth_rate = 24
                dense_n = 3
                knn = 16
                use_bn = False
                bn_decay = None
                comp = growth_rate * 2
                # features = ops.mlp_conv(inputs, [self.start_number, self.start_number * 2])
                # features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
                # features = tf.concat(
                #     [features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1, 1])],
                #     axis=-1
                # )
                l0_features = ops.conv2d(
                    inputs, 24, [1, 1],
                    padding='VALID', scope='layer0', is_training=self.is_training, bn=self.bn,
                    bn_decay=0.95, activation_fn=None
                )
                l0_features = tf.squeeze(l0_features, axis=2)

                l1_features, l1_idx = ops.dense_conv1(
                    l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                    scope='layer1', is_training=self.is_training, bn=use_bn, bn_decay=bn_decay
                )
                out_feat = tf.concat([l1_features, l0_features], axis=-1)
                features_gloabl = tf.reduce_max(out_feat, axis=1, keep_dims=True, name='maxpool_0')
                features = tf.concat(
                    [out_feat, tf.tile(features_gloabl, [1, tf.shape(inputs)[1], 1, 1])]
                )

                # l2_features = ops.conv1d(
                #     out_feat, comp, 1,
                #     padding='VALID', scope='layer2_prep', is_training=self.is_training, bn=use_bn,
                #     bn_decay=bn_decay
                # )
                # l2_features, l2_idx = ops.dense_conv1(
                #     l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                #     scope="layer2", is_training=self.is_training, bn=use_bn, bn_decay=bn_decay
                # )
                # out_feat = tf.concat([l2_features, out_feat], axis=-1)

                # out_feat = tf.expand_dims(out_feat, axis=2)
                # features = ops.attention_unit(out_feat, is_training=self.is_training)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
                features = ops.mlp_conv(features, [self.start_number * 4, self.start_number * 8])
                features = tf.reduce_max(features, axis=1, name='maxpool_1')

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                outputs = ops.mlp(features, [self.start_number * 8, 1])
                outputs = tf.reshape(outputs, [-1, 1])

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return outputs