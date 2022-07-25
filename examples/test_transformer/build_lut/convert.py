from nn_meter.builder.backends import connect_backend
from nn_meter.builder import convert_models, builder_config
workspace = "/data/data0/jiahang/tflite_space"
builder_config.init(workspace)

backend = connect_backend("tflite_cpu")

convert_models(backend, models=f"/data/data0/jiahang/vit_lut/results/lut_v1.json", broken_point_mode=True)
