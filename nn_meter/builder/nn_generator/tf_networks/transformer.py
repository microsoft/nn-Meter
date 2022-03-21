import tensorflow as tf
from tensorflow import keras
from torch import softmax
from .operators import *
LAYER_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


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
    attn = Softmax(axis=-1)(attn)

    # output = tf.matmul(attn, vv)    # [batch, num_heads, q_blocks, key_dim * attn_ratio]
    output = Matmul([attn, vv])
    output = tf.transpose(output, perm=[0, 2, 1, 3], name=name and name + "transpose")  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2] * output.shape[3]])  # [batch, q_blocks, channel * attn_ratio]
    output = Hswish(output)
    output = keras.layers.Dense(output_dim, use_bias=False, name=name + "out")(output)
    if not layer_norm:
        output = BatchNorm(output)
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
    qkv = BatchNorm(qkv)
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

    inputs = LayerNorm(inputs, name=name+'_ln_attn')
    if window_size > 1:
        ww = window_size * window_size
        inputs = tf.reshape(inputs, (-1, ww, C))
        _B, _, _ = inputs.shape
    else:
        ww = blocks
        _B = B
    
    qkv_dim = int(2 * embed_dim + embed_dim_v)
    qkv = keras.layers.Dense(qkv_dim, use_bias=False, name=name + "qkv")(inputs)
    qkv = Hswish(qkv)
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
    nn = BatchNorm(nn)
    nn = Hswish(nn)
    nn = keras.layers.Dense(in_channels, use_bias=use_bias)(nn)
    nn = BatchNorm(nn)
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return Add([inputs, nn])


def res_mlp_block_layer_norm(inputs, mlp_ratio, drop_rate=0, use_bias=False, activation="hard_swish", name=""):
    # MLP layernorm
    in_channels = inputs.shape[-1]
    inputs = LayerNorm(inputs)
    nn = keras.layers.Dense(int(in_channels * mlp_ratio), use_bias=use_bias)(inputs)
    nn = Hswish(nn)
    nn = keras.layers.Dense(in_channels, use_bias=use_bias)(nn)
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return Add(name=name + "add")([inputs, nn])
