import os
import tensorflow as tf
from tensorflow.keras import layers
from nn_meter.predictor import load_latency_predictor
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import builder_config
from nn_meter.builder.nn_generator.tf_networks.utils import get_inputs_by_shapes
from nn_meter.dataset.bench_dataset import latency_metrics

from nas_models.blocks.tf.mobilenetv3_block import HSigmoid
from nas_models.common import make_divisible

workspace = "/data1/jiahang/working/pixel4_mobilenetv3_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')


output_path = "/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor"
output_name = os.path.join(output_path, "MobilenetV3_test")
predictor_name = "tflite27_cpu_int8"
predictor = load_latency_predictor(predictor_name)


def profile_and_predict(model, input_shape, name="se"):
    print("\n")
    # print(model)
    # input_shape example [224, 224, 3]
    model(get_inputs_by_shapes([[*input_shape]]))
    tf.keras.models.save_model(model, output_name)

    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
 
    # pred_lat = predictor.predict(model, "torch", input_shape=tuple([1] + input_shape), apply_nni=False) # in unit of ms
    pred_lat = sum(predictor.kernel_predictors[name].predict([[input_shape[0], input_shape[-1]]])) # in unit of ms
    # print("profiled: ", res["latency"].avg, "predicted: ", pred_lat, input_shape[0], input_shape[-1])
    input_shape = list(model(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
    return res["latency"].avg, pred_lat


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

# configs = [
#     # HW, CIN
#     [112, 16], [28, 120], [14, 120], [14, 480], [14, 480],
#     [14, 240], [14, 240], [14, 320], [14, 320], [14, 672],
#     [14, 672], [14, 448], [14, 448], [14, 336], [14, 336],
#     [14, 672], [7, 672], [7, 640], [7, 640], [7, 480],
#     [7, 480], [7, 960], [1, 1280],
# ]
# models = [HSwish_NNMETER, HSwish_OFA]

# reals, preds = [], []
# for model_cls in models:
#     for i, config in enumerate(configs):
#         hwin, cin = config
#         input_shape = [hwin, hwin, cin]
#         model = model_cls()
#         real, pred = profile_and_predict(model, input_shape, 'hswish')
#         reals.append(real)
#         preds.append(pred)

#     rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
#     for item in zip(preds, reals):
#         print(item)
#     print(f"[Hswish] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")


configs = [
    # HW, CIN
    [28, 72], [28, 160], [14, 320], [14, 672], 
	[14, 448], [14, 336], [7, 672], [7, 640], [7, 480],
    [1, 96], [1, 160], [1, 240], [1, 240], [1, 448],
    [1, 672], [1, 672], [1, 672], [1, 960], [1, 480]
]
models = [SE_NNMETER, SE_OFA]

reals, preds = [], []
for model_cls in models:
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
# SE_OFA:
# SE_NNMETER: