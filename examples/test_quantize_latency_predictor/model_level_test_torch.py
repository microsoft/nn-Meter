import os
import json
import time
import torch
from torch import nn
# from tensorflow import keras
from nn_meter.dataset.bench_dataset import latency_metrics
from nn_meter.builder.backends import connect_backend
from nn_meter.predictor import load_latency_predictor
from nn_meter.builder import builder_config
from nn_meter.builder.nn_generator.torch_networks.utils import get_inputs_by_shapes

from nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
from nas_models.blocks.torch.mobilenetv3_block import SE


output_path = "/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor"
output_name = os.path.join(output_path, "MobilenetV3_test.onnx")

workspace = "/sdc/jiahang/working/ort_mobilenetv3_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='ort_cpu_int8')
predictor_name = "onnxruntime_int8"
predictor = load_latency_predictor(predictor_name)


def profile_and_predict(model, input_shape, mark = "", model_pred = None):
    # print("\n")
    # print(model)
    # input_shape example [3, 224, 224]
    torch.onnx.export(
            model,
            get_inputs_by_shapes([[*input_shape]], 1),
            output_name,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
        )
    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
    if model_pred != None:
        pred_lat = predictor.predict(model_pred, "torch", input_shape=tuple([1] + input_shape), apply_nni=False) # in unit of ms
    else:
        pred_lat = predictor.predict(model, "torch", input_shape=tuple([1] + input_shape), apply_nni=False) # in unit of ms
    # print(f"[{mark}]: ", "profiled: ", res["latency"].avg, "predicted: ", pred_lat)
    input_shape = list(model(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
    return res["latency"].avg, pred_lat

## ------------- model level
sample_str = "ks55355773757755735757_e66643464363346436436_d22343"
def get_model_result(sample_str):
    model = MobileNetV3Net(sample_str)
    real, pred = profile_and_predict(model, [3, 224, 224], mark="")
    print("profiled: ", real, "predicted: ", pred)
    return real, pred

## ------------- block level
def get_torch_blocks(sample_str):
    from nas_models.blocks.torch.mobilenetv3_block import block_dict, BasicBlock
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
            print(hwin, cin, cout, k, strides)
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
    input_shape = [3, 224, 224]
    for i, block in enumerate(get_torch_blocks(sample_str)):
        real, pred = profile_and_predict(block, input_shape, mark=str(i))      
        input_shape = list(block(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
        real_collection.append(real)
        pred_collection.append(pred)
    return sum(real_collection), sum(pred_collection)


def model_level_test():
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
        # print(pred_collection, pred)
        assert int(pred) == int(pred_collection)
        model_res.append(real)
        blocks_res.append(real_collection)
        pred_res.append(pred)

    print(model_res)
    print(blocks_res)
    print(pred_res)

    from nn_meter.dataset.bench_dataset import latency_metrics
    # [12.249987502582371, 10.697516263462603, 15.283254371024668, 12.977135782130063, 15.040200399234891, 16.805960782803595, 12.333641056902707, 12.11845989804715, 10.843409495428205, 9.86639161594212, 10.824401904828846, 8.921326529234648, 12.164815440773964, 16.722270911559463, 13.340506758540869, 11.978719434700906, 10.842160694301128, 13.073826101608574, 13.135424223728478, 13.751582726836205, 12.951741521246731, 11.028801458887756, 14.39663636032492, 12.55843972787261, 12.064272919669747, 13.979026176966727, 14.876921479590237, 13.215321684256196, 19.26913076546043, 12.96588427387178]
    # [13.174293381161988, 12.25960579700768, 17.13364808820188, 14.944951985962689, 17.91127840988338, 17.945631546899676, 13.884303728118539, 13.503874726593494, 12.722029406577349, 11.452783001586795, 12.793498081155121, 10.536506134085357, 14.052079082466662, 16.62240343634039, 14.206458372063935, 14.229186330921948, 12.51209313981235, 14.730160813778639, 13.773103589192033, 14.5341558707878, 14.315508562140167, 12.821959354914725, 16.43644588533789, 14.975599735043943, 13.641426763497293, 16.015005614608526, 16.31128814537078, 15.228850464336574, 18.268028628081083, 15.642286906950176]
    # (1.7215729352537374, 11.978833812405412, 0.11829990383432135, 0.06666666666666667, 0.3, 0.8)
    print(latency_metrics(model_res, blocks_res))


if __name__ == '__main__':
    model_level_test()
