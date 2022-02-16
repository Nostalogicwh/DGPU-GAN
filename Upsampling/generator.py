import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample


class Generator(object):
    def __init__(self, opts, is_training, name='Generator'):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # 特征提取
            # 思路 更换特征提取组件
            features = ops.feature_extraction(
                inputs,
                scope='feature_extraction',
                is_training=self.is_training,
                bn_decay=None
            )

            # 上采样
            # 思路 上下上是否必要
            h = ops.up_projection_unit(
                features,
                self.up_ratio_real,
                scope='up_projection_unit',
                is_training=self.is_training,
                bn_decay=None
            )
            # 坐标重建
            coord = ops.conv2d(
                h, 64, [1, 1],
                padding='VALID',
                stride=[1, 1],
                bn=False,
                is_training=self.is_training,
                scope='fc_layer1',
                bn_decay=None
            )

            coord = ops.conv2d(
                coord, 3, [1, 1],
                padding='VALID',
                stride=[1, 1],
                bn=False,
                is_training=self.is_training,
                scope='fc_layer2',
                bn_decay=None,
                activation_fn=None,
                weight_decay=0.0
            )

            # 最远点采样
            outputs = tf.squeeze(coord, [2])
            outputs = gather_point(
                outputs,
                farthest_point_sample(
                    self.out_num_point,
                    outputs
                )
            )
        self.reuse = True
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            self.name
        )
        return outputs

