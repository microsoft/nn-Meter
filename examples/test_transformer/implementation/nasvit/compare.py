import json

with open("/data/data0/jiahang/tflite_space/predictor_build/results_pixel6/nasvit_lut_v1.json", "r") as fp:
    nasvit_space = json.load(fp)
with open("/data/data0/jiahang/tflite_space/predictor_build/results_pixel6/nasvit_lut_ln_v3.json", "r") as fp:
    extend_sapce = json.load(fp)["lut"]
print(len(extend_sapce))
for item in nasvit_space:
    if item.startswith("MBlock"):
        key = item.replace("MBlock_", "conv_")
        if key not in extend_sapce:
            print("!!!", key)
    elif item.startswith("Trans_"):
        if "nods" in item:
            name, hw, cin, cout, exp, s, act, v, ds, ln = item.split("_")
            keys = [
                f"nasvit_transattn_{hw}_{cin}_{act}_{v}_{ln}",
                f"nasvit_transffn_{hw}_{cin}_{exp}_{act}_{ln}"
            ]
            for key in keys:
                if key not in extend_sapce:
                    print("!!!", item)
                    break
        else:
            name, hw, cin, cout, exp, s, act, v, ds, ds_exp, ln = item.split("_")
            keys = [f"nasvit_nose_transds_{hw}_{cin}_{cout}_{s}_{ds_exp}"]
            if s == "2":
                hw = int(hw) // 2
            keys.append(f"nasvit_transattn_{hw}_{cout}_{act}_{v}_{ln}")
            keys.append(f"nasvit_transffn_{hw}_{cout}_{exp}_{act}_{ln}")
            for key in keys:
                # import pdb; pdb.set_trace()
                if key not in extend_sapce:
                    if "nose" in key and key.replace("nose", "se") not in extend_sapce:
                        print("!!!", item, key)
                    if "nose" not in key:
                        print("!!!", item, key)
                
                    break

        