import os, json
import random
import string
import tensorflow as tf
import tensorflow.keras as keras
from conv_block import conv_layer
from transformer_block import transformer_layer
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"

'''
sample = (
    176, # input res
    (0, 0, 0, 1, 1, 1), # block type
    (24, 40, 80, 112, 192, 320), # channels
    (2, 3, 2, 2, 3, 4), # depths
    (6, 6, 6, 6, 6, 4, 4), # conv expansion ratio
    (3, 3, 3, 3, 3, 5, 5), # conv kr size
    (2, 2, 4, 4, 4, 4, 4, 4, 4), # trans mlp ratio
    (14, 14, 24, 24, 24, 40, 40, 40, 40), # trans num heads
    (1, 1, 1, 1, 1, 1, 1, 1, 1), # windows size
    (1, 1, 1, 1, 1, 1, 1, 1, 1), # qk scale
    (2, 2, 2, 2, 2, 2, 2, 2, 2) # v scale
)
'''

def sampler(configs):
    block_type = []
    channels = []
    depths = []
    conv_exp = []
    conv_ks = []
    trans_exp = []
    trans_numheads = []
    window_size = []
    qk_scale = []
    v_scale = []

    resolution = random.choice(configs[0]['hw'])

    for i, config in enumerate(configs[1:]):
        block_type.append(config['block_type'])
        depth = random.choice(config['depth'])
        channel = random.choice(config['channel'])
        channels.append(channel)
        depths.append(depth)
        if config['block_type'] == 0:
            for _ in range(depth):
                exp = random.choice(config['expansion_ratio'])
                ks = random.choice(config['kernel size'])
                conv_exp.append(exp)
                conv_ks.append(ks)
        else:
            for _ in range(depth):
                exp = random.choice(config['expansion_ratio'])
                v = random.choice(config['v_scale'])
                trans_numheads.append(channel // 8)
                window_size.append(1)
                qk_scale.append(1)
                trans_exp.append(exp)
                v_scale.append(v)
    sample = (
        resolution,
        block_type,
        channels,
        depths,
        conv_exp,
        conv_ks,
        trans_exp,
        trans_numheads,
        window_size,
        qk_scale,
        v_scale
    )
    return sample


def model_builder(block_config):
    # f'conv_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}'
    # f'transformer_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}'
    strides = [2, 2, 2, 1, 2, 2]
    hw = block_config[0]
    inputs = keras.Input(shape=[hw, hw, 3], batch_size=1)

    # first_block
    nn = conv_layer(inputs, 16, 1, 3, 2)

    # layer_choice blocks
    conv_count, trans_count = 0, 0
    for stage_idx, block_type in enumerate(block_config[1]):
        name = "conv" if block_type == 0 else "transformer"
        stage_stride = strides[stage_idx]
        stage_hwout = hw // stage_stride if hw % stage_stride == 0 else hw // stage_stride + 1
        hw = stage_hwout
        stage_cout = block_config[2][stage_idx]
        if name == "conv":
            for i in range(block_config[3][stage_idx]):
                s = stage_stride if i == 0 else 1
                cout = stage_cout
                exp = block_config[4][conv_count]
                ks = block_config[5][conv_count]
                nn = conv_layer(nn, cout, exp, ks, s)
                conv_count += 1

        elif name == "transformer":
            for i in range(block_config[3][stage_idx]):
                s = stage_stride if i == 0 else 1
                cout = stage_cout
                exp = block_config[6][trans_count]
                v = block_config[10][trans_count]
                nn = transformer_layer(nn, cout, exp, v, s, i, name=f'trans{trans_count}')
                trans_count += 1

    assert conv_count == len(block_config[5])
    assert trans_count == len(block_config[6])
    
    # print("here", nn.shape)
    model = keras.Model(inputs=inputs, outputs=nn)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    converted_model = os.path.join(main_path, "models", f"{random_id}.tflite")
    open(converted_model, 'wb').write(tflite_model)
    return random_id

from space_utils import configs as c3t3_config
from space_utils_2c4t import configs as c2t4_config
from space_utils_4c2t import configs as c4t2_config

from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor
predictor = BlockLatencyPredictor("mobile_lut")

res = {}
for configs in [c2t4_config, c4t2_config]:
    for i in range(500):
        sample = sampler(c3t3_config)
        # print(sample)
        id = model_builder(sample)
        pred_lat = predictor.get_latency(sample)
        res[id] = {
            "converted_model": f"/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/kernels/{id}.tflite",
            'shapes': [[sample[0], sample[0], 3]],
            'configs': sample,
            'pred_lat': pred_lat
            }
        if i % 50 == 0:
            with open(os.path.join(main_path, "results", f"model_v2.json"), 'w') as fp:
                json.dump({"lut": res}, fp)

with open(os.path.join(main_path, "results", f"model_v2.json"), 'w') as fp:
    json.dump({"lut": res}, fp)

# nohup python /data/data0/jiahang/vit_lut/build_model.py > trans_model_log.txt 2>&1 &
# [1] 52002