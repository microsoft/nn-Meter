import os
from tensorflow import keras
import tensorflow as tf
from nn_meter.dataset.bench_dataset import latency_metrics
from nn_meter.builder.backends import connect_backend
from nn_meter.predictor import load_latency_predictor
from nn_meter.builder import builder_config
from nn_meter.builder.nn_generator.tf_networks.utils import get_inputs_by_shapes

from nas_models.blocks.torch.mobilenetv3_block import block_dict, BasicBlock
from nas_models.search_space.mobilenetv3_space import MobileNetV3Space
from nas_models.common import parse_sample_str
output_path = "/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor"
output_name = os.path.join(output_path, "MobilenetV3_test")

workspace = "/data1/jiahang/working/pixel4_mobilenetv3_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')
predictor_name = "tflite27_cpu_int8"

# workspace = "/data1/jiahang/working/tflite21_workspace"
# builder_config.init(workspace)
# backend = connect_backend(backend_name='tflite_cpu')
# predictor_name = "cortexA76cpu_tflite21"

predictor = load_latency_predictor(predictor_name)
def profile_and_predict(model_tf, model_torch, input_shape):
    print("\n")
    model_tf(get_inputs_by_shapes([[*input_shape]]))
    tf.keras.models.save_model(model_tf, output_name)

    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])

    pred_lat = predictor.predict(model_torch, "torch", input_shape=tuple([1] + [input_shape[2], input_shape[0], input_shape[1]]), apply_nni=False) # in unit of ms
    # print(pred_lat)
    print("profiled: ", res["latency"].avg, "predicted: ", pred_lat)
    return res["latency"].avg, pred_lat

# ## ------------- model level
sample_str = "ks55355773757755735757_e66643464363346436436_d22343"
def get_model_result(sample_str):
    from nas_models.networks.tf.mobilenetv3 import MobileNetV3Net as MobileNetV3Net_tf
    from nas_models.networks.torch.mobilenetv3 import MobileNetV3Net as MobileNetV3Net_torch
    model_tf = MobileNetV3Net_tf(sample_str)
    model_torch = MobileNetV3Net_torch(sample_str)
    real, pred = profile_and_predict(model_tf, model_torch, [224, 224, 3])
    return real, pred

# ##------- block level random sample
# ##--- first mobile conv
# from nas_models.blocks.tf.mobilenetv3_block import FirstConv as FirstConv_tf
# from nas_models.blocks.torch.mobilenetv3_block import FirstConv as FirstConv_torch
# model_tf = FirstConv_tf(224, 3, 16)
# model_torch = FirstConv_torch(224, 3, 16)
# print(model_torch)
# predictor = load_latency_predictor(predictor_name)
# real, pred = profile_and_predict(model_tf, model_torch, [224, 224, 3])

# ##--- first mobile conv
# from nas_models.blocks.tf.mobilenetv3_block import FirstMBConv as FirstMBConv_tf
# from nas_models.blocks.torch.mobilenetv3_block import FirstMBConv as FirstMBConv_torch
# model_tf = FirstMBConv_tf(112, 16, 16)
# model_torch = FirstMBConv_torch(112, 16, 16)
# print(model_torch)
# predictor = load_latency_predictor(predictor_name)
# real, pred = profile_and_predict(model_tf, model_torch, [112, 112, 16])


# # ##--- RandomMBConv1
# # input:28x28x40-output:28x28x40-k:3-e:6-act:relu-se:1
# from nas_models.blocks.tf.mobilenetv3_block import MBConv as MBConv_tf
# from nas_models.blocks.torch.mobilenetv3_block import MBConv as MBConv_torch
# model_tf = MBConv_tf(28, 40, 40, 3, 6, 1, act='relu', se=0)
# model_torch = MBConv_torch(28, 40, 40, 3, 6, 1, act='relu', se=0)
# print(model_torch)
# predictor = load_latency_predictor(predictor_name)
# real, pred = profile_and_predict(model_tf, model_torch, [28, 28, 40])

# # RandomMBConv2
# # input:56x56x24-output:56x56x24-k:3-e:3-act:relu-se:0
# from nas_models.blocks.tf.mobilenetv3_block import MBConv as MBConv_tf
# from nas_models.blocks.torch.mobilenetv3_block import MBConv as MBConv_torch
# model_tf = MBConv_tf(56, 24, 24, 3, 3, 2, act='relu', se=0)
# model_torch = MBConv_torch(56, 24, 24, 3, 3, 2, act='h_swish', se=0)
# print(model_torch)
# predictor = load_latency_predictor(predictor_name)
# real, pred = profile_and_predict(model_tf, model_torch, [56, 56, 24])


# ##------- block level lut
def get_tf_blocks(sample_str):
    from nas_models.search_space.mobilenetv3_space import MobileNetV3Space
    from nas_models.blocks.tf.mobilenetv3_block import FirstConv, FirstMBConv, MBConv, FinalExpand, FeatureMix, Logits
    width_mult = 1.0
    num_classes = 1000
    hw = 224
    space = MobileNetV3Space(width_mult=width_mult, num_classes=num_classes, hw=hw)

    sample_config = parse_sample_str(sample_str)

    blocks = []
    first_conv = FirstConv(hwin=hw, cin=3, cout=space.stage_width[0])
    first_mbconv = FirstMBConv(
        hwin=hw//2,
        cin=space.stage_width[0],
        cout=space.stage_width[1]
    )
    blocks.append(first_conv)
    blocks.append(first_mbconv)

    hwin = hw // 2
    cin = space.stage_width[1]
    block_idx = 0
    for strides, cout, max_depth, depth, act, se in zip(
        space.stride_stages[1:], space.stage_width[2:], 
        space.num_block_stages[1:], sample_config['d'],
        space.act_stages[1:], space.se_stages[1:]
    ):
        for i in range(depth):
            k = sample_config['ks'][block_idx + i]
            e = sample_config['e'][block_idx + i]
            strides = 1 if i > 0 else strides
            blocks.append(MBConv(hwin, cin, cout, kernel_size=k, expand_ratio=e, strides=strides,
                act=act, se=int(se)))
            cin = cout 
            hwin //= strides
        block_idx += max_depth
    # blocks = tf.keras.Sequential(blocks)

    final_expand = FinalExpand.build_from_config(space.block_configs[-3])
    blocks.append(final_expand)
    feature_mix = FeatureMix.build_from_config(space.block_configs[-2])
    blocks.append(feature_mix)
    logits = Logits.build_from_config(space.block_configs[-1])
    blocks.append(logits)
    return blocks


def get_torch_blocks(sample_str):
    from nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
    from nas_models.blocks.torch.mobilenetv3_block import SE, block_dict, BasicBlock
    from nas_models.search_space.mobilenetv3_space import MobileNetV3Space
    from nas_models.common import parse_sample_str
    
    width_mult = 1.0
    num_classes = 1000
    hw = 224
    space = MobileNetV3Space(width_mult=width_mult, num_classes=num_classes, hw=hw)

    sample_config = parse_sample_str(sample_str)

    blocks = []
    first_conv = block_dict['first_conv'](hwin=hw, cin=3, cout=space.stage_width[0])
    first_mbconv = block_dict['first_mbconv'](
        hwin=hw//2,
        cin=space.stage_width[0],
        cout=space.stage_width[1]
    )
    blocks.append(first_conv)
    blocks.append(first_mbconv)

    hwin = hw // 2
    cin = space.stage_width[1]
    block_idx = 0
    for strides, cout, max_depth, depth, act, se in zip(
        space.stride_stages[1:], space.stage_width[2:], 
        space.num_block_stages[1:], sample_config['d'],
        space.act_stages[1:], space.se_stages[1:]
    ):
        for i in range(depth):
            k = sample_config['ks'][block_idx + i]
            e = sample_config['e'][block_idx + i]
            strides = 1 if i > 0 else strides
            # print(hwin, cin, cout, k, strides)
            blocks.append(block_dict['mbconv'](hwin, cin, cout, kernel_size=k, expand_ratio=e, strides=strides,
                act=act, se=int(se)))
            cin = cout 
            hwin //= strides
        block_idx += max_depth
    # blocks = nn.Sequential(*blocks)

    final_expand = block_dict['final_expand'].build_from_config(space.block_configs[-3])
    blocks.append(final_expand)
    feature_mix = block_dict['feature_mix'].build_from_config(space.block_configs[-2])
    blocks.append(feature_mix)
    logits = block_dict['logits'].build_from_config(space.block_configs[-1])
    blocks.append(logits)
    return blocks


def get_block_result(sample_str):
    real_collection, pred_collection = [], []
    input_shape = [224, 224, 3]
    for block_tf, block_torch in zip(get_tf_blocks(sample_str), get_torch_blocks(sample_str)):
        real, pred = profile_and_predict(block_tf, block_torch, input_shape)      
        input_shape = list(block_tf(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
        real_collection.append(real)
        pred_collection.append(pred)
        print("Complete one block !!!!")
        # break

    # print(real_collection, pred_collection)
    return sum(real_collection), sum(pred_collection)

def model_level_test_mobilenetv3():
    sample_strs = [
        "ks33575373355333733735_e36436643443366644444_d34224",
        "ks35557755553357557577_e34444634446634344346_d32422",
        "ks35573553777537353577_e66663643664464634434_d34434",
        "ks35575755357375333535_e34666346446634666633_d33243",
        "ks35733755577537357533_e43646344343444466666_d44433",
        "ks37757555775335335773_e64343634434466646363_d43234",
        "ks37773735737755577555_e44643443334363446366_d32423",
        "ks53553373573735557757_e34636464334363443346_d42324",
        "ks53555333377537333333_e33633333633343636334_d32432",
        "ks55553557735733337735_e36464663366646633664_d22224",
        "ks55555733737375537357_e34644636646344434364_d24323",
        "ks55573357337355575377_e33343463434364663346_d22422",
        "ks55737375555373577575_e36446363464646643466_d23242",
        "ks55753555755337375337_e66344643364433434463_d34343",
        "ks57335333733333533377_e33346433336463334364_d43234",
        "ks57337553533753375775_e44663363644434663633_d33233",
        "ks57375737773337373753_e33364336363434633364_d22332",
        "ks57557335355733777337_e63444464363664336346_d23344",
        "ks57733337777753577735_e63646443644334363433_d24234",
        "ks73733375355333755335_e33344646466636636466_d23434",
        "ks75337773755575777735_e36364334333636463364_d32433",
        "ks75355577775533333577_e33463434364443336334_d32433",
        "ks75373355357337757553_e43436636646446446663_d34342",
        "ks75577553557535557753_e46434336343336466343_d32342",
        "ks77333575553355355757_e44663344344363346444_d24423",
        "ks77533353535353555375_e43344646446663636433_d43243",
        "ks77533573753375577735_e64334433446646446333_d43422",
        "ks77575333733335375335_e44364434346443664444_d44322",
        "ks77773333355577337577_e33336464633644643333_d43434",
        "ks77773533335735575575_e66466646643433364334_d24243"
    ]
    model_res, blocks_res = [], []
    pred_res = []
    for sample_str in sample_strs:
        real, pred = get_model_result(sample_str)
        real_collection, pred_collection = get_block_result(sample_str)
        print(pred_collection, pred)
        assert int(pred) == int(pred_collection)
        model_res.append(real)
        blocks_res.append(real_collection)
        pred_res.append(pred)
        break
    # print(model_res)
    # print(blocks_res)
    # print(pred_res)

    # 30个模型，profiled model latency和sum of blocks latency均在5%误差内
    from nn_meter.dataset.bench_dataset import latency_metrics
    print(latency_metrics(model_res, blocks_res))


if __name__ == '__main__':
    model_level_test_mobilenetv3()