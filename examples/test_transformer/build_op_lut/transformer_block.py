import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.download_and_load import reload_model_weights_with_mismatch
from keras_cv_attention_models.attention_layers import depthwise_conv2d_no_bias, batchnorm_with_activation, conv2d_no_bias, activation_by_name,layer_norm
# from .space_utils import ACT
ACT = 'hard_swish'

def attention_downsampling(inputs, out_channels, downsampling, dwconv=False, exp=6, use_se=False):
    if len(inputs.shape) == 3:
        B, L, C = inputs.shape
        # print(inputs.shape)
        H = W = int(L**0.5)
        inputs = tf.reshape(inputs, [-1, H, W, C])
    
    if downsampling:
        nn = conv2d_no_bias(inputs, out_channels, 3, strides=2, padding='same', use_bias=False)
    else:
        nn = conv2d_no_bias(inputs, out_channels, 1, strides=1, padding='same', use_bias=False)
    nn = batchnorm_with_activation(nn, activation=ACT)
   
    B, H, W, C = nn.shape
    nn = tf.reshape(nn, [-1, int(H*W), out_channels])
    return nn

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

def scaled_dot_product_attention(qq, kk, vv, key_dim, attn_ratio, output_dim, activation="hard_swish", name="", layer_norm=False, talking_head=False):
    # qq, kk, vv: [batch, num_heads, blocks, key_dim]
    FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    qk_scale = tf.math.sqrt(tf.cast(key_dim, FLOAT_DTYPE))
    # print(f"{qq.shape = }, {kk.shape = }")
    # attn = tf.matmul(qq, kk, transpose_b=True) / qk_scale   # [batch, num_heads, q_blocks, k_blocks]
    attn = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True), name=name and name + "Lambda")([qq, kk]) / qk_scale
    # print(f"{attn.shape = }")
    attn = MultiHeadPositionalEmbedding(name=name + "attn_pos")(attn)
    # attn = tf.nn.softmax(attn, axis=-1)
    
    if talking_head:
        # [B, heads, ww, dim]
        heads = attn.shape[1]
        attn = tf.transpose(attn, perm=[0, 2, 3, 1], name=name and name + "_reshape_pre_attn_proj_1")
        attn = keras.layers.Dense(heads, use_bias=False, name=name + "pre_attn_proj")(attn)
        attn = tf.transpose(attn, perm=[0, 3, 1, 2], name=name and name + "_reshape_pre_attn_proj_2")
    
    attn = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    if talking_head:
        attn = tf.transpose(attn, perm=[0, 2, 3, 1], name=name and name + "_reshape_after_attn_proj_1")
        attn = keras.layers.Dense(heads, use_bias=False, name=name + "after_attn_proj")(attn)
        attn = tf.transpose(attn, perm=[0, 3, 1, 2], name=name and name + "_reshape_after_attn_proj_2")

    # output = tf.matmul(attn, vv)    # [batch, num_heads, q_blocks, key_dim * attn_ratio]
    output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]), name=name and name + "Lambda2")([attn, vv])
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!", attn.shape, vv.shape, output.shape)
    # exit()
    output = tf.transpose(output, perm=[0, 2, 1, 3], name=name and name + "transpose")  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2] * output.shape[3]], name=name and name + "reshape")  # [batch, q_blocks, channel * attn_ratio]
    if activation:
        output = activation_by_name(output, activation=activation, name=name)
    output = keras.layers.Dense(output_dim, use_bias=False, name=name + "out")(output)
    if not layer_norm:
        output = batchnorm_with_activation(output, activation=None, zero_gamma=True, name=name + "out_")
    return output

def attention(inputs, output_dim, num_heads, key_dim, v_dim, window_size, activation="hard_swish", name=""):
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

    vv = tf.reshape(vv, (-1, int(ww**0.5), int(ww**0.5), embed_dim_v))
    vv = depthwise_conv2d_no_bias(vv, kernel_size=3, strides=1, padding='same')
    vv = tf.reshape(vv, (-1, num_heads, ww, v_dim))
    
    att = scaled_dot_product_attention(qq, kk, vv, key_dim, 1, output_dim=output_dim, activation=activation, name=name+'attn', talking_head=True)
    att = tf.reshape(att, (-1, blocks, C))
    return att

def mlp(inputs, mlp_ratio, drop_rate=0, use_bias=False, activation="hard_swish", name=""):
    in_channels = inputs.shape[-1]

    nn = keras.layers.Dense(int(in_channels * mlp_ratio), use_bias=use_bias, name=name + "1_dense")(inputs)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = keras.layers.Dense(in_channels, use_bias=use_bias, name=name + "2_dense")(nn)
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return keras.layers.Add(name=name + "add")([inputs, nn])

# please note that only the ***first layer*** in a transformer stage will have a downsample layer
def transformer_layer(inputs, channels, expansion_ratio, v_scale, stride, layer_index, name):
    if layer_index == 0:
        inputs = attention_downsampling(inputs, channels, downsampling=stride==2, dwconv=True)

    head_dim = 8
    num_heads = channels//head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim*v_scale

    nn = attention(inputs, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+'att')
    nn = mlp(nn, expansion_ratio, activation=ACT, name=name+'mlp')

    return nn


def trans_ops(inputs, channels, expansion_ratio, v_scale, stride, layer_index, name):
    log = {}
    if layer_index == 0:
        input_shape = inputs.shape
        inputs = attention_downsampling(inputs, channels, downsampling=stride==2, dwconv=True)
        log["attention_downsampling"] = [list(input_shape), list(inputs.shape), channels, stride==2, True]

    head_dim = 8
    num_heads = channels//head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim*v_scale

    input_shape = inputs.shape
    nn = attention(inputs, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+'att')
    log["attention"] = [list(input_shape), list(nn.shape), channels, num_heads, key_dim, v_dim, 1, ACT]
    
    input_shape = nn.shape
    nn = mlp(nn, expansion_ratio, activation=ACT, name=name+'mlp')
    log["mlp"] = [list(input_shape), list(nn.shape), expansion_ratio, ACT]
    

    return log