import os, json

predictor_name = "pixel4_lut"
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "lut", f"{predictor_name}_ln_v2.json"), 'r') as fp:
    predictor = json.load(fp)

for item in predictor:
    if item.startswith("transformer") and item.endswith("ln"):
        if "nods" in item:
            name, hw, cin, cout, exp, s, act1, act2, v, ds, ln = item.split("_")
            lpy = predictor[item]
            tpy = predictor[f"transattn_{hw}_{cin}_{act1}_{act2}_{v}_{ln}"] + \
                predictor[f"transffn_{hw}_{cin}_{exp}_{act1}_{act2}_{ln}"]
            
        else:
            name, hw, cin, cout, exp, s, act1, act2, v, ds, ds_exp, ln = item.split("_")
            # import pdb; pdb.set_trace()
            tpy = predictor[f"transds_{hw}_{cin}_{cout}_{s}_{ds_exp}"]
            if s == "2":
                hw = int(hw) // 2
            lpy = predictor[item]
            tpy += predictor[f"transattn_{hw}_{cout}_{act1}_{act2}_{v}_{ln}"] + \
                predictor[f"transffn_{hw}_{cout}_{exp}_{act1}_{act2}_{ln}"]
        open("/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/compare.txt", "a").write(f"{item}, {lpy}, {tpy}\n")