from nn_meter.builder.nn_generator.tf_networks.operators import *

def scaled_dot_product_attention(qq, kk, vv, key_dim, attn_ratio, output_dim):
    # qq, kk, vv: [batch, num_heads, blocks, key_dim]
    
    # FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    # qk_scale = tf.math.sqrt(tf.cast(key_dim, FLOAT_DTYPE))
    qk_scale = ...
    
    # print(f"{qq.shape = }, {kk.shape = }")
    # attn = tf.matmul(qq, kk, transpose_b=True) / qk_scale   # [batch, num_heads, q_blocks, k_blocks]
    attn = Matmul([qq, kk]) / qk_scale # [[4, 4, 49, 32], [4, 4, 49, 32]] -> [4, 4, 49, 49]
    # print(f"{attn.shape = }")
    attn = MultiHeadPositionalEmbedding(attn) # [4, 4, 49, 49] -> [4, 4, 49, 49]
    # attn = tf.nn.softmax(attn, axis=-1)
    attn = Softmax(axis=-1)(attn) # [4, 4, 49, 49] -> [4, 4, 49, 49]

    # output = tf.matmul(attn, vv)    # [batch, num_heads, q_blocks, key_dim * attn_ratio]
    output = Matmul([attn, vv]) # [[4, 4, 49, 49], [4, 4, 49, 32]] -> [4, 4, 49, 32]
    # [batch, q_blocks, num_heads, key_dim * attn_ratio]
    output = Transpose(perm=[0, 2, 1, 3])(output)  # [4, 4, 49, 32] -> [4, 49, 4, 32]
    # [batch, q_blocks, channel * attn_ratio]
    output = Reshape([-1, output.shape[1], output.shape[2] * output.shape[3]])(output) # [4, 49, 4, 32] -> [4, 49, 128]
    output = Hswish(output) # [4, 49, 128] -> [4, 49, 128]
    output = FC(output_dim, use_bias=False)(output) # [4, 49, 128] -> [4, 49, 128]
    output = BatchNorm(output) # [4, 49, 128] -> [4, 49, 128]
    return output


def mhsa_with_multi_head_position_windows(inputs, output_dim, num_heads, key_dim, v_dim, window_size):
    # attention, batchnorm
    B, blocks, C = inputs.shape # [1, 196, 128]
    key_dim = int(key_dim) # 32
    embed_dim = key_dim * num_heads # 128
    embed_dim_v = int(v_dim * num_heads) # 128

    if window_size > 1:
        ww = window_size * window_size # 49
        inputs = Reshape((-1, ww, C))(inputs) # [1, 14, 14, 128] -> [4, 49, 128]
        _B, _, _ = inputs.shape # [4, 49, 128]
    else:
        ww = blocks
        _B = B

    qkv_dim = int(2 * embed_dim + embed_dim_v) # 384
    qkv = FC(qkv_dim, use_bias=False)(inputs) # [4, 49, 128] -> [4, 49, 384]
    qkv = BatchNorm(qkv) # [4, 49, 384] -> [4, 49, 384]
    qkv = Reshape((-1, ww, num_heads, int(qkv_dim//num_heads)))(qkv) # [4, 49, 384] -> [4, 49, 4, 96]
    qkv = Transpose(perm=[0, 2, 1, 3])(qkv) # [4, 49, 4, 96] -> [4, 4, 49, 96]
    qq, kk, vv = Split([key_dim, key_dim, v_dim])(qkv) # [4, 4, 49, 96] -> [[4, 4, 49, 32], [4, 4, 49, 32], [4, 4, 49, 32]]
    att = scaled_dot_product_attention(qq, kk, vv, key_dim, 1, output_dim=output_dim) # [4, 4, 49, 32] -> [4, 49, 128]
    att = Reshape((-1, blocks, C))(att) # [4, 4, 49, 32] -> [1, 196, 128]
    return att


def mhsa_with_multi_head_position_windows_layer_norm(inputs, output_dim, num_heads, key_dim, v_dim, window_size):
    # attention layernorm
    B, blocks, C = inputs.shape # [1, 14, 14, 128]
    key_dim = int(key_dim) # 32
    embed_dim = key_dim * num_heads # 128
    embed_dim_v = int(v_dim * num_heads) # 128

    inputs = LayerNorm(inputs) # [1, 14, 14, 128] -> [1, 14, 14, 128]
    if window_size > 1:
        ww = window_size * window_size # 49
        inputs = Reshape((-1, ww, C))(inputs) # [1, 14, 14, 128] -> [4, 49, 128]
        _B, _, _ = inputs.shape # [4, 49, 128]
    else:
        ww = blocks
        _B = B
    
    qkv_dim = int(2 * embed_dim + embed_dim_v) # 384
    qkv = FC(qkv_dim, use_bias=False)(inputs) # [4, 49, 128] -> [4, 49, 384]
    # qkv = Hswish(qkv) # [4, 49, 384] -> [4, 49, 384]
    qkv = Reshape((-1, ww, num_heads, int(qkv_dim//num_heads)))(qkv) # [4, 49, 384] -> [4, 49, 4, 96]
    qkv = Transpose(perm=[0, 2, 1, 3])(qkv) # [4, 49, 4, 96] -> [4, 4, 49, 96]
    qq, kk, vv = Split([key_dim, key_dim, v_dim])(qkv) # [4, 4, 49, 96] -> [[4, 4, 49, 32], [4, 4, 49, 32], [4, 4, 49, 32]]
    att = scaled_dot_product_attention(qq, kk, vv, key_dim, 1, output_dim=output_dim, layer_norm=True) # [4, 4, 49, 32] -> [4, 49, 128]
    att = Reshape((-1, blocks, C))(att) # [4, 4, 49, 32] -> [1, 196, 128]
    return att


def res_mlp_block(inputs, mlp_ratio, drop_rate=0, use_bias=False):
    # MLP batchnorm
    in_channels = inputs.shape[-1] # [1, 14, 14, 128]

    nn = FC(int(in_channels * mlp_ratio))(inputs)  # [1, 14, 14, 128] -> [1, 14, 14, 256]
    nn = BatchNorm(nn) # [1, 14, 14, 256] -> [1, 14, 14, 256]
    nn = Hswish(nn) # [1, 14, 14, 256] -> [1, 14, 14, 256]
    nn = FC(in_channels)(nn) # [1, 14, 14, 256] -> [1, 14, 14, 128]
    nn = BatchNorm(nn) # [1, 14, 14, 128] -> [1, 14, 14, 128]
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1))(nn) # [1, 14, 14, 128] -> [1, 14, 14, 128]
    return Add([inputs, nn]) # [1, 14, 14, 128] -> [1, 14, 14, 128]


def res_mlp_block_layer_norm(inputs, mlp_ratio, drop_rate=0, use_bias=False):
    # MLP layernorm
    in_channels = inputs.shape[-1] # [1, 14, 14, 128]
    inputs = LayerNorm(inputs) # [1, 14, 14, 128] -> [1, 14, 14, 128]
    nn = FC(int(in_channels * mlp_ratio))(inputs) # [1, 14, 14, 128] -> [1, 14, 14, 256]
    nn = Hswish(nn) # [1, 14, 14, 256] -> [1, 14, 14, 256]
    nn = FC(in_channels)(nn) # [1, 14, 14, 256] -> [1, 14, 14, 128]
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1))(nn) # [1, 14, 14, 128] -> [1, 14, 14, 128]
    return Add([inputs, nn]) # [1, 14, 14, 128] -> [1, 14, 14, 128]
