"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import numpy as np
import tensorflow as tf

from config import cfg


epsilon = 1e-9


class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    # 输出是一个向量（例如，一个8维的向量）。每个向量的长度代表了物体是否存在的估计概率，它的方向（例如在8维空间里）
    # 记录了物体的姿态参数（比如，精确的位置、旋转等）。如果物体有稍微的变化（比如，移动、旋转、尺寸变化等），
    # 胶囊将也会输出一个长度相同但是方向稍微变化的向量。因此胶囊是等变的。

    # 初始化，初始化胶囊的个数，每个胶囊向量的长度，每层的类型
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs      # 胶囊个数
        self.vec_len = vec_len              # 每个胶囊向量的长度
        self.with_routing = with_routing
        self.layer_type = layer_type

    # 定义卷积层的输出
    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        input: 卷积层的输出
        '''
        # 卷积层的类型为CNN，定义了卷积核的步长，卷积核的大小
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            # Conv 1拥有256个9×9卷积核，步长为1，ReLU激活。
            # 该图层将像素转换为局部特征的激活图，然后将其用作主胶囊（primary capsule）的输入
            # 从Conv 1层到PrimaryCaps layer层，不进行动态路由。将Conv 1层的输出当作PrimaryCaps layer层的输入
            if not self.with_routing:
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                # Conv 1层长为20，宽为256，高为20
                assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

                '''
                # version 1, computational expensive
                capsules = []
                for i in range(self.vec_len):
                    # each capsule i: [batch_size, 6, 6, 32]
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID", activation_fn=None)
                        caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                        capsules.append(caps_i)
                assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]
                capsules = assert tf.concat(capsules, axis=2)
                '''

                # version 2, equivalent to version 1 but higher computational
                # efficiency.
                # NOTE: I can't find out any words from the paper whether the
                # PrimaryCap convolution does a ReLU activation or not before
                # squashing function, but experiment show that using ReLU get a
                # higher test accuracy. So, which one to use will be your choice

                # 第二层（PrimaryCapsules）是具有32个卷积8D胶囊通道的卷积胶囊层（即，每个主胶囊包含8个卷积单元，9×9内核和步长为2）
                # 每个主胶囊单元的值，是通过计算Conv 1中所有256×81单位的输出，其感受野与胶囊中心的位置重叠。
                # PrimaryCapsules有[32×6×6]胶囊输出（每个输出是一个8D矢量），[6×6]网格中的每个胶囊彼此共享它们的权重。
                # 第二层（PrimaryCapsules）将胶囊看成是一个像素点，则为6*6*32

                # PrimaryCaps layer层的输入为Conv 1层的输出，其中input为input.get_shape() == [cfg.batch_size, 20, 20, 256]
                # self.num_outputs=32，self.vec_len=8，self.kernel_size=9，self.stride=2
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)
                # capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                #                                    self.kernel_size, self.stride,padding="VALID",
                #                                    activation_fn=None)
                # self.vec_len为向量的长度，等于8
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

                # [batch_size, 1152, 8, 1]
                # 将胶囊压缩到[0,1]之间，squash为定义的压缩函数
                capsules = squash(capsules)
                # DigitCaps的第一层有1152个向量，每个向量为8D，纬度为1
                #  最后一层（DigitCaps）每个类有一个16 D胶囊用来进行表示，每个胶囊都接收来自上层级中中所有胶囊的输出作为输入。
                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                # 返回胶囊
                return(capsules)

        # 只在两个连续胶囊层之间进行路由（例如PrimaryCapsules和DigitCaps）
        # 由于Conv 1的输出是1D，因此在其空间中没有方向可以达成一致。 因此，Conv 1和PrimaryCapsules之间不使用路由。
        if self.layer_type == 'FC':
            # 层与层之间使用动态路由
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    # np.zeros生成5个元素（类型为float）的矩阵
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    # 对全连接层进行动态路由
                    capsules = routing(self.input, b_IJ)
                    # tf.squeeze()函数的作用是从tensor中删除所有大小(szie)是1的维度。
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


# 定义动态路由
def routing(input, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''
    # 输入是[batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]，num_caps_l是第一层胶囊的个数，向量ui的长度为8

    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    # num_caps_i=1代表第i层胶囊的个数，num_caps_j=1152，len_u_i=8代表向量ui的长度，len_v_j=16代表向量vj的长度
    # 随机初始化权重wij
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    # tf.tile对纬度进行扩张
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    # u_hat代表uj|i，uj|i=Wij*ui
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    # 路由算法第3行的实现，迭代3次
    for r_iter in range(cfg.iter_routing):
        # 返回一个用于定义创建variable（层）的op的上下文管理器
        # 该上下文管理器验证（可选）值来自同一图形，确保图形是默认图形，并推送名称范围和variable范围。
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            # 路由算法第4行，对bij进行softmax
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


# 定义squash压缩函数，将向量的值限制在[0,1]的范围内
def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
