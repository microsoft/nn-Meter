from nn_meter.predictor import load_latency_predictor
import json
from .nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
from nn_meter.builder.nn_modules.torch_networks.utils import get_inputs_by_shapes
from nn_meter.dataset.bench_dataset import latency_metrics
from nn_meter.builder.backend_meta.utils import Latency

predictor_name = "tflite27_cpu_int8"
info_save_path = "/data1/jiahang/working/pixel4_mobilenetv3_workspace/predictor_build/results/profiled_mobilenetv3_3.json"

predictor = load_latency_predictor(predictor_name)

with open(info_save_path, 'r') as fp:
    profiling_list = json.load(fp)

# predict latency
result = {}
True_lat, Pred_lat = [], []
for _, module in profiling_list.items():
    for config, model_info in module.items():
        print(config)
        model = MobileNetV3Net(config)
        pred_lat = predictor.predict(model, "torch", input_shape=(1, 3, 224, 224), apply_nni=False) # in unit of ms
        real_lat = Latency(model_info["latency"]).avg
        model_info["predict_latency"] = pred_lat
        print(f"#####: pred: {pred_lat}, real: {real_lat}")
        if real_lat != None:
            True_lat.append(float(real_lat))
            Pred_lat.append(pred_lat)
        # break

if len(True_lat) > 0:
    rmse, rmspe, error, acc5, acc10, _ = latency_metrics(Pred_lat, True_lat)
    print(f"rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}")

with open(info_save_path, 'w') as fp:
    json.dump(profiling_list, fp, indent=4)
