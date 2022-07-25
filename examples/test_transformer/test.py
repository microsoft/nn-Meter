import json
with open("/data1/jiahang/working/pixel6_fp32_workspace/nn-Meter/nn_meter/predictor/transformer_predictor/pixel6_lut.json", "r") as fp:
    res = json.load(fp)
print(len(res))

res_new = {}
for k in res.keys():
    k_new = k
    if k.startswith("conv_") and (not (k.endswith("True") or k.endswith("False"))):
        k_new += "_False"
    if k_new not in res_new:
        res_new[k_new] = res[k]
    else:
        print(k, (res_new[k_new]- res[k])/res_new[k_new])
    # res_new
print(len(res_new))

with open("/data1/jiahang/working/pixel6_fp32_workspace/nn-Meter/nn_meter/predictor/transformer_predictor/pixel6_lut.json", 'w') as fp:
    json.dump(res_new, fp, indent=4)
