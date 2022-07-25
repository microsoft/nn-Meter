import json

def collect_exist_lut():
    with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_v3.json", 'r') as fp:
        lut_result_ref = json.load(fp)
    with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_v4.json", 'r') as fp:
        lut_result_ref.update(json.load(fp))
    with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_v5.json", 'r') as fp:
        lut_result_ref.update(json.load(fp))
    print(len(lut_result_ref))
    
    with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_v5_all.json", 'r') as fp:
        lut_result = json.load(fp)

    # lut_result = {}
    # for key in lut_result_ref:
    #     if key.startswith("transformer") and key.endswith("ds"):
    #         lut_result[key] = lut_result_ref[key]

    for key in lut_result:
        lut_result[key] = lut_result_ref[key]

    with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_v5_all.json", 'w') as fp:
        json.dump(lut_result, fp, indent=4)


collect_exist_lut()