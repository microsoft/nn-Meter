import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from nn_meter.predictor import load_latency_predictor
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import builder_config
from nn_meter.builder.nn_generator.tf_networks.utils import get_inputs_by_shapes
from nn_meter.predictor.prediction.utils import latency_metrics_acc20 as latency_metrics
from nn_meter.builder.kernel_predictor_builder.predictor_builder.utils import get_flops_params

from nas_models.blocks.tf.mobilenetv3_block import HSigmoid
from nas_models.common import make_divisible

workspace = "/data1/jiahang/working/pixel4_mobilenetv3_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu')


output_path = "/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor"
output_name = os.path.join(output_path, "MobilenetV3_test")
predictor_name = "tflite27_cpu_int8"
# predictor_name = "cortexA76cpu_tflite21"
predictor = load_latency_predictor(predictor_name)


def profile_and_predict(model, input_shape, name="se"):
    # print("\n")
    # print(model)
    # input_shape example [224, 224, 3]
    model(get_inputs_by_shapes([[*input_shape]]))
    tf.keras.models.save_model(model, output_name)

    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
 
    # pred_lat = predictor.predict(model, "torch", input_shape=tuple([1] + input_shape), apply_nni=False) # in unit of ms
    pred_lat = sum(predictor.kernel_predictors[name].predict([[input_shape[0], input_shape[-1]]])) # in unit of ms
    print("profiled: ", res["latency"].avg, "predicted: ", pred_lat)
    # input_shape = list(model(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
    return res["latency"].avg, pred_lat


def profile_model(model, input_shape):
    # print("\n")
    # print(model)
    model(get_inputs_by_shapes([[*input_shape]]))
    tf.keras.models.save_model(model, output_name)

    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])

    return res["latency"].avg


class HSwish_NNMETER(tf.keras.Model):
    
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.nn.relu6(tf.math.add(x, 3)) * 0.16667


class HSwish_OFA(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)


class SE_OFA(tf.keras.Model):

    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.squeeze = layers.Conv2D(filters=make_divisible(num_channels * se_ratio), kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.excite = layers.Conv2D(filters=num_channels, kernel_size=1, padding='same')
        self.hsigmoid = HSigmoid()

    def call(self, x):
        x0 = x
        x = self.pool(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hsigmoid(x)
        return x * x0
    

class SE_NNMETER(tf.keras.Model):
    def __init__(self, cin, hw):
        super().__init__()
        self.cin = cin
        self.hw = hw
        self.conv1 = tf.keras.layers.Conv2D(
            filters=cin // 4,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=cin,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )

    def call(self, inputs):
        # hw = inputs.shape[1]
        x = tf.nn.avg_pool(
            inputs,
            ksize=[1, self.hw, self.hw, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu6(tf.math.add(x, 3)) * 0.16667
        return x * inputs

def compare_op_hswish():
    configs = [
        # HW, CIN
        [112, 16], [28, 120], [14, 120], [14, 480], [14, 480],
        [14, 240], [14, 240], [14, 320], [14, 320], [14, 672],
        [14, 672], [14, 448], [14, 448], [14, 336], [14, 336],
        [14, 672], [7, 672], [7, 640], [7, 640], [7, 480],
        [7, 480], [7, 960], [1, 1280],
    ]
    models = [HSwish_NNMETER, HSwish_OFA]

    for model_cls in models:
        reals, preds = [], []
        for i, config in enumerate(configs):
            hwin, cin = config
            input_shape = [hwin, hwin, cin]
            model = model_cls()
            real, pred = profile_and_predict(model, input_shape, 'hswish')
            reals.append(real)
            preds.append(pred)

        rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
        for item in zip(preds, reals):
            print(item)
        print(f"[Hswish] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")
    # [Hswish(nn-meter)] rmse: 0.005325419195994526, rmspe: 128.7342249029692, error: 0.03191439393194509, acc5: 0.9565217391304348, acc10: 0.9565217391304348, acc15: 0.9565217391304348
    # [Hswish(Xudong version)] rmse: 0.14610096071155015, rmspe: 611.0362854025838, error: 3.26282225890695, acc5: 0.0, acc10: 0.0, acc15: 0.0


def compare_op_se():
    configs = [
        # HW, CIN
        [28, 72], [28, 160], [14, 320], [14, 672],
        [14, 448], [14, 336], [7, 672], [7, 640], [7, 480]
    ]
    models = [SE_OFA]

    for model_cls in models:
        reals, preds = [], []
        for i, config in enumerate(configs):
            hwin, cin = config
            input_shape = [hwin, hwin, cin]
            model = model_cls(cin, hwin)
            real, pred = profile_and_predict(model, input_shape, 'se')
            reals.append(real)
            preds.append(pred)

        rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
        for item in zip(preds, reals):
            print(item)
        print(f"[SE] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")

    # SE_NNMETER: [SE] rmse: 0.01494647811039525, rmspe: 12.331616679372031, error: 0.14495697356533918, acc5: 0.5555555555555556, acc10: 0.5555555555555556, acc15: 0.6666666666666666
    # SE_OFA: [SE] rmse: 2.0208344320323324, rmspe: 93.8414818102769, error: 1.0535283572775551, acc5: 0.0, acc10: 0.0, acc15: 0.0


def compare_op_dwconv():
    from tensorflow import keras
    from nas_models.blocks.tf.mobilenetv3_block import build_act
    class DwconvTest(tf.keras.Model):
        def __init__(self, kernel_size, strides, act):
            super().__init__()
            self.depth_conv = tf.keras.Sequential([
                        keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same'),
                        keras.layers.BatchNormalization(),
                        build_act(act)
                    ])
        def call(self, x):
            x = self.depth_conv(x)
            return x

    def profile_and_predict(model, input_shape, name="se"):
        print("\n")
        # print(model)
        # input_shape example [224, 224, 3]
        model(get_inputs_by_shapes([[*input_shape]]))
        tf.keras.models.save_model(model, output_name)

        res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
    
        # pred_lat = predictor.predict(model, "torch", input_shape=tuple([1] + input_shape), apply_nni=False) # in unit of ms
        pred_lat = sum(predictor.kernel_predictors[name].predict([[28, 240, 240, 3, 1, 1.8816, 0.0024]])) # in unit of ms
        print("profiled: ", res["latency"].avg, "predicted: ", pred_lat)
        # input_shape = list(model(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
        return res["latency"].avg, pred_lat
    predictor = load_latency_predictor(predictor_name)
    model = DwconvTest(3, 1, 'relu')
    # model(get_inputs_by_shapes([[28, 28, 240]], batch_size=1))
    profile_and_predict(model, [28, 28, 240], 'dwconv-bn-relu')


def get_feature(kernel_type, config_dict):
    needed_config = {
        "conv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
        "dwconv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
        "se": ["HW", "CIN"],
        "hswish": ["HW", "CIN"],
    }
    if "COUT" not in config_dict and "COUT" in needed_config[kernel_type]:
        config_dict["COUT"] = config_dict["CIN"]
    feature = [config_dict[data] for data in needed_config[kernel_type]]
    if kernel_type in ["conv-bn-relu", "dwconv-bn-relu"]:
        flop, param = get_flops_params(kernel_type, config_dict)
        flop /= 2e6
        param /= 1e6
        feature.extend([flop, param])
    return feature

## ------------- op level
from nn_meter.builder.nn_generator.tf_networks.blocks import ConvBnRelu, DwConvBnRelu, HswishBlock, SEBlock

def op_level_test_conv(predictor_name):
    # conv-bn-relu
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)

    reals, preds = [], []
    configs = [
        # mobilenet v3
        [224, 3, 16, 3, 2], [56, 48, 24, 1, 1], [56, 24, 144, 1, 1], [56, 144, 24, 1, 1], [56, 24, 96, 1, 1], [56, 96, 24, 1, 1],
        [28, 144, 40, 1, 1], [28, 40, 240, 1, 1], [28, 240, 40, 1, 1], [28, 40, 160, 1, 1], [28, 160, 40, 1, 1], [28, 40, 120, 1, 1],
        [28, 120, 40, 1, 1], [14, 160, 80, 1, 1], [14, 80, 320, 1, 1], [14, 320, 80, 1, 1], [14, 80, 480, 1, 1], [14, 480, 112, 1, 1],
        [14, 112, 672, 1, 1], [14, 672, 112, 1, 1], [14, 112, 448, 1, 1], [7, 448, 160, 1, 1], [7, 160, 640, 1, 1], [7, 640, 160, 1, 1],
        [7, 160, 960, 1, 1], [1, 960, 1280, 1, 1], [28, 96, 40, 1, 1], [14, 480, 80, 1, 1], [14, 80, 240, 1, 1], [14, 240, 112, 1, 1],
        [14, 448, 112, 1, 1], [7, 160, 480, 1, 1], [7, 480, 160, 1, 1], [112, 16, 96, 1, 1], [56, 24, 72, 1, 1], [28, 72, 40, 1, 1], 
        [14, 240, 80, 1, 1], [7, 672, 160, 1, 1], [7, 960, 160, 1, 1], [112, 16, 64, 1, 1], [56, 64, 24, 1, 1], [56, 72, 24, 1, 1], 
        [14, 120, 80, 1, 1], [14, 320, 112, 1, 1], [14, 112, 336, 1, 1], [14, 336, 112, 1, 1], [7, 336, 160, 1, 1]
        # resnet
        
    ]
    # for i, config in enumerate(configs):
    for i, cout in enumerate(range(600, 681)):
        # hwin, cin, cout, k, strides = config
        hwin, cin, cout, k, strides = 56, 640, cout, 1, 1
        config_in = {
            "HW": hwin,
            "CIN": cin,
            "COUT": cout,
            "KERNEL_SIZE": k,
            "STRIDES": strides
        }
        input_shape = [hwin, hwin, cin]
        model = ConvBnRelu(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("conv-bn-relu", config_in)])[0]
        reals.append(real)
        preds.append(pred)

    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    # for item in zip(reals, preds):
    #     open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_conv.txt", "a").write(f'{item}\n')
    # open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_conv.txt", "a").write(f"[Conv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}\n")
    for cin, res in zip(range(600, 681), reals):
        open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_conv_cinrange.txt", "a").write(f"cin: {cin}; profiled results: {res}\n")
    open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_conv_cinrange.txt", "a").write(f"[Conv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}\n")
    

def op_level_test_dwconv(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)
    # dwconv-bn-relu
    reals, preds = [], []
    configs = [
        [112, 16, 3, 1], [112, 48, 3, 2], [56, 144, 3, 1], [56, 96, 5, 1], [56, 144, 5, 2], [28, 240, 3, 1], [28, 160, 7, 1],
        [28, 120, 3, 1], [28, 160, 3, 2], [14, 320, 5, 1], [14, 480, 3, 1], [14, 672, 3, 1], [14, 448, 3, 2], [7, 640, 7, 1],
        [7, 640, 3, 1], [7, 640, 5, 1], [56, 96, 7, 2], [28, 240, 7, 1], [28, 160, 5, 2], [14, 240, 5, 1], [14, 448, 7, 1],
        [14, 448, 7, 2], [7, 480, 5, 1], [112, 96, 3, 2], [56, 144, 5, 1], [56, 72, 3, 2], [28, 240, 5, 1], [28, 160, 5, 1],
        [28, 240, 7, 2], [14, 480, 7, 1], [14, 320, 7, 1], [7, 480, 7, 1], [28, 120, 7, 1], [14, 240, 7, 1], [14, 448, 5, 1],
        [14, 672, 3, 2], [7, 960, 5, 1], [7, 480, 3, 1], [112, 64, 3, 2], [56, 72, 5, 1], [56, 144, 7, 1], [56, 96, 3, 1],
        [56, 144, 3, 2], [28, 120, 5, 2], [14, 320, 3, 1], [14, 448, 3, 1], [14, 672, 7, 2], [7, 960, 3, 1], [56, 96, 7, 1],
        [56, 72, 7, 1], [56, 72, 7, 2], [28, 120, 5, 1], [28, 160, 7, 2], [14, 672, 5, 1], [14, 672, 5, 2], [7, 960, 7, 1],
        [28, 120, 7, 2], [14, 240, 3, 1], [14, 480, 5, 1], [14, 336, 5, 1], [112, 48, 5, 2], [28, 160, 3, 1], [14, 336, 7, 2],
        [56, 72, 3, 1], [56, 72, 5, 2], [28, 240, 3, 2], [14, 336, 7, 1], [56, 96, 3, 2], [56, 96, 5, 2], [14, 336, 5, 2],
        [56, 144, 7, 2], [112, 96, 5, 2], [14, 448, 5, 2], [14, 336, 3, 1], [112, 64, 5, 2], [28, 240, 5, 2], [14, 336, 3, 2],
        [28, 120, 3, 2], [112, 48, 7, 2], [14, 672, 7, 1], [112, 64, 7, 2], [112, 96, 7, 2]
    ]
    # for i, config in enumerate(configs):
    for i, cin in enumerate(range(600, 681)):
        # hwin, cin, k, strides = config
        hwin, cin, k, strides = 112, cin, 3, 1
        config_in = {
            "HW": hwin,
            "CIN": cin,
            "COUT": cin,
            "KERNEL_SIZE": k,
            "STRIDES": strides
        }
        input_shape = [hwin, hwin, cin]
        model = DwConvBnRelu(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("dwconv-bn-relu", config_in)])[0]
        reals.append(real)
        preds.append(pred)
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for cin, res in zip(range(600, 681), reals):
        open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_dwconv_cinrange.txt", "a").write(f"cin: {cin}; profiled results: {res}\n")
    open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_dwconv_cinrange.txt", "a").write(f"[Dwconv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}\n")
    # for item in zip(reals, preds):
    #     open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_dwconv.txt", "a").write(f'{item}\n')
    # open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_dwconv.txt", "a").write(f"[Dwconv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}\n")


def op_level_test_hswish(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)

    reals, preds = [], []
    configs = [
        [112, 16], [28, 120], [14, 120], [14, 480], [14, 480], [14, 240], [14, 240], [14, 320],
        [14, 320], [14, 672], [14, 672], [14, 448], [14, 448], [14, 336], [14, 336], [14, 672],
        [7, 672], [7, 640], [7, 640], [7, 480], [7, 480], [7, 960], [1, 1280],
    ]

    # for i, config in enumerate(configs):
    for i, cin in enumerate(range(600, 681)):
        # hwin, cin = config
        hwin, cin = 14, cin
        config_in = {
            "HW": hwin,
            "CIN": cin
        }
        input_shape = [hwin, hwin, cin]
        model = HSwish_OFA()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("hswish", config_in)])[0]
        reals.append(real)
        preds.append(pred)
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for cin, res in zip(range(600, 681), reals):
        open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_hswish.txt", "a").write(f"cin: {cin}; profiled results: {res}\n")
    # for item in zip(reals, preds):
    #     open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_hswish.txt", "a").write(f'{item}')
    open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_hswish.txt", "a").write(f"[Hswish] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


def op_level_test_se(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)
    # se

    reals, preds = [], []
    configs = [
        [28, 72], [28, 160], [14, 320], [14, 672], [14, 448], [14, 336], 
        [7, 672], [7, 640], [7, 480], [112, 16], [28, 120], [14, 120], [14, 480], [14, 480], [14, 240], [14, 240], [14, 320],
        [14, 320], [14, 672], [14, 672], [14, 448], [14, 448], [14, 336], [14, 336], [14, 672],
        [7, 672], [7, 640], [7, 640], [7, 480], [7, 480], [7, 960]
    ]
    # for i, config in enumerate(configs):
    for cin in range(600, 681):
        # hwin, cin = config
        hwin, cin = 14, cin
        config_in = {
            "HW": hwin,
            "CIN": cin
        }
        input_shape = [hwin, hwin, cin]
        model = SEBlock(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("se", config_in)])[0]
        reals.append(real)
        preds.append(pred)
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for cin, res in zip(range(600, 681), reals):
        open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_se.txt", "a").write(f"cin: {cin}; profiled results: {res}\n")
    # for item in zip(reals, preds):
    #     open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_se.txt", "a").write(f'{item}')
    open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_se.txt", "a").write(f"[SE] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")

if __name__ == '__main__':
    
    # op_level_test_conv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/conv-bn-relu_original.pkl")
    # op_level_test_conv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/conv-bn-relu_ofa.pkl")
    # op_level_test_conv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/conv-bn-relu_ofa_only.pkl")
    # op_level_test_conv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/conv-bn-relu_ofa_filt8.pkl")
    
    # op_level_test_dwconv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_original.pkl")
    # op_level_test_dwconv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa.pkl")
    # op_level_test_dwconv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa_only.pkl")
    # op_level_test_dwconv("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa_filt8.pkl")
    
    # op_level_test_hswish("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/hswish_prior.pkl")
    op_level_test_hswish("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/hswish_ofa.pkl")
    # op_level_test_hswish("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/hswish_ofa_only.pkl")

    # op_level_test_se("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/se_prior.pkl")
    # op_level_test_se("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/se_ofa.pkl")
    # op_level_test_se("/data1/jiahang/working/pixel4_int8_workspace/predictor_build/results/predictors/se_ofa_only.pkl")
    
