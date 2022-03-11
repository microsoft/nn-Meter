import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

LAYER_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


@tf.keras.utils.register_keras_serializable(package="common")
def hard_swish(inputs):
    """ `out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244 """
    return inputs * tf.nn.relu6(inputs + 3) / 6


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=None):
    """ Performs a batch normalization followed by an activation. """
    bn_axis = 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=0.9,
        epsilon=1e-5,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def activation_by_name(inputs, activation="relu", name=None):
    """ Typical Activation layer added hard_swish and prelu. """
    layer_name = name and activation and name + activation
    if activation == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation.lower() == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(0) # channel axes
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)
    else:
        return inputs


def layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=None):
    """ Typical LayerNormalization with epsilon=1e-5 """
    norm_axis = 1 # channel axes
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=LAYER_NORM_EPSILON, name=name and name + "ln")(inputs)


@tf.keras.utils.register_keras_serializable(package="levit")
class MultiHeadPositionalEmbedding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MultiHeadPositionalEmbedding, self).__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        _, num_heads, qq_blocks, kk_blocks = input_shape
        self.bb = self.add_weight(name="positional_embedding", shape=(kk_blocks, num_heads), initializer="zeros", trainable=True)
        strides = int(tf.math.ceil(tf.math.sqrt(float(kk_blocks / qq_blocks))))
        q_blocks_h = q_blocks_w = int(tf.math.sqrt(float(qq_blocks)))
        k_blocks_h = k_blocks_w = int(tf.math.sqrt(float(kk_blocks)))

        x1, y1 = tf.meshgrid(range(q_blocks_h), range(q_blocks_w))
        x2, y2 = tf.meshgrid(range(k_blocks_h), range(k_blocks_w))
        aa = tf.concat([tf.reshape(x1, (-1, 1)), tf.reshape(y1, (-1, 1))], axis=-1)
        bb = tf.concat([tf.reshape(x2, (-1, 1)), tf.reshape(y2, (-1, 1))], axis=-1)
        # print(f">>>> {aa.shape = }, {bb.shape = }") # aa.shape = (16, 2), bb.shape = (49, 2)
        cc = [tf.math.abs(bb - ii * strides) for ii in aa]
        self.bb_pos = tf.stack([ii[:, 0] + ii[:, 1] * k_blocks_h for ii in cc])
        # print(f">>>> {self.bb_pos.shape = }")    # self.bb_pos.shape = (16, 49)

        super(MultiHeadPositionalEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        pos_bias = tf.gather(self.bb, self.bb_pos)
        pos_bias = tf.transpose(pos_bias, [2, 0, 1])
        return inputs + pos_bias

    def load_resized_pos_emb(self, source_layer):
        if isinstance(source_layer, dict):
            source_bb = source_layer["positional_embedding:0"]  # weights
        else:
            source_bb = source_layer.bb  # layer
        hh = ww = int(tf.math.sqrt(float(source_bb.shape[0])))
        ss = tf.reshape(source_bb, (hh, ww, source_bb.shape[-1]))  # [hh, ww, num_heads]
        target_hh = target_ww = int(tf.math.sqrt(float(self.bb.shape[0])))
        tt = tf.image.resize(ss, [target_hh, target_ww])  # [target_hh, target_ww, num_heads]
        tt = tf.reshape(tt, (self.bb.shape))  # [target_hh * target_ww, num_heads]
        self.bb.assign(tt)


def scaled_dot_product_attention(qq, kk, vv, key_dim, attn_ratio, output_dim, activation="hard_swish", name="", layer_norm=False):
    # qq, kk, vv: [batch, num_heads, blocks, key_dim]
    FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    qk_scale = tf.math.sqrt(tf.cast(key_dim, FLOAT_DTYPE))
    # print(f"{qq.shape = }, {kk.shape = }")
    # attn = tf.matmul(qq, kk, transpose_b=True) / qk_scale   # [batch, num_heads, q_blocks, k_blocks]
    attn = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True), name=name and name + "Lambda")([qq, kk]) / qk_scale
    # print(f"{attn.shape = }")
    attn = MultiHeadPositionalEmbedding(name=name + "attn_pos")(attn)
    # attn = tf.nn.softmax(attn, axis=-1)
    attn = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    # output = tf.matmul(attn, vv)    # [batch, num_heads, q_blocks, key_dim * attn_ratio]
    output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]), name=name and name + "Lambda2")([attn, vv])
    output = tf.transpose(output, perm=[0, 2, 1, 3], name=name and name + "transpose")  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2] * output.shape[3]], name=name and name + "reshape")  # [batch, q_blocks, channel * attn_ratio]
    if activation:
        output = activation_by_name(output, activation=activation, name=name)
    output = keras.layers.Dense(output_dim, use_bias=False, name=name + "out")(output)
    if not layer_norm:
        output = batchnorm_with_activation(output, activation=None, zero_gamma=True, name=name + "out_")
    return output


def mhsa_with_multi_head_position_windows(inputs, output_dim, num_heads, key_dim, v_dim, window_size, activation="hard_swish", name=""):
    # attention, batchnorm
    B, blocks, C = inputs.shape
    key_dim = int(key_dim)
    embed_dim = key_dim * num_heads
    embed_dim_v = int(v_dim * num_heads)

    if window_size > 1:
        ww = window_size * window_size
        inputs = tf.reshape(inputs, (-1, ww, C))
        _B, _, _ = inputs.shape
    else:
        ww = blocks
        _B = B

    qkv_dim = int(2 * embed_dim + embed_dim_v)
    qkv = keras.layers.Dense(qkv_dim, use_bias=False, name=name + "qkv")(inputs)
    qkv = batchnorm_with_activation(qkv, activation=None, name=name + "qkv_")
    qkv = tf.reshape(qkv, (-1, ww, num_heads, int(qkv_dim//num_heads)), name=name and name + "reshape")
    qkv = tf.transpose(qkv, perm=[0, 2, 1, 3], name=name and name + "Lambda")
    qq, kk, vv = tf.split(qkv, [key_dim, key_dim, v_dim], axis=-1, name=name and name + "split")
    att = scaled_dot_product_attention(qq, kk, vv, key_dim, 1, output_dim=output_dim, activation=activation, name=name)
    att = tf.reshape(att, (-1, blocks, C))
    return att


def mhsa_with_multi_head_position_windows_layer_norm(inputs, output_dim, num_heads, key_dim, v_dim, window_size, activation="hard_swish", name=""):
    # attention layernorm
    B, blocks, C = inputs.shape
    key_dim = int(key_dim)
    embed_dim = key_dim * num_heads
    embed_dim_v = int(v_dim * num_heads)

    inputs = layer_norm(inputs, name=name+'_ln_attn')
    if window_size > 1:
        ww = window_size * window_size
        inputs = tf.reshape(inputs, (-1, ww, C))
        _B, _, _ = inputs.shape
    else:
        ww = blocks
        _B = B
    
    qkv_dim = int(2 * embed_dim + embed_dim_v)
    qkv = keras.layers.Dense(qkv_dim, use_bias=False, name=name + "qkv")(inputs)
    qkv = activation_by_name(qkv, activation=activation, name=name+'_act')
    qkv = tf.reshape(qkv, (-1, ww, num_heads, int(qkv_dim//num_heads)), name=name and name + "reshape")
    qkv = tf.transpose(qkv, perm=[0, 2, 1, 3], name=name and name + "Lambda")
    qq, kk, vv = tf.split(qkv, [key_dim, key_dim, v_dim], axis=-1, name=name and name + "split")
    att = scaled_dot_product_attention(qq, kk, vv, key_dim, 1, output_dim=output_dim, activation=activation, name=name, layer_norm=True)
    att = tf.reshape(att, (-1, blocks, C))
    return att


def res_mlp_block(inputs, mlp_ratio, drop_rate=0, use_bias=False, activation="hard_swish", name=""):
    # MLP batchnorm
    in_channels = inputs.shape[-1]

    nn = keras.layers.Dense(int(in_channels * mlp_ratio), use_bias=use_bias, name=name + "1_dense")(inputs)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = keras.layers.Dense(in_channels, use_bias=use_bias, name=name + "2_dense")(nn)
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return keras.layers.Add(name=name + "add")([inputs, nn])


def res_mlp_block_layer_norm(inputs, mlp_ratio, drop_rate=0, use_bias=False, activation="hard_swish", name=""):
    # MLP layernorm
    in_channels = inputs.shape[-1]
    inputs = layer_norm(inputs, name=name+'_ln')
    nn = keras.layers.Dense(int(in_channels * mlp_ratio), use_bias=use_bias, name=name + "1_dense")(inputs)
    nn = activation_by_name(nn, activation=activation, name=name+'_act')
    nn = keras.layers.Dense(in_channels, use_bias=use_bias, name=name + "2_dense")(nn)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return keras.layers.Add(name=name + "add")([inputs, nn])

