import pickle
with open("/data/jiahang/working/nnmeterdata/predictor/adreno630gpu_tflite21/conv-bn-relu.pkl", "rb") as f:
    predictor = pickle.load(f)

print(predictor.predict([[1,2,3,4]]))
print(predictor.predict([[1,2,3,]]))
print(predictor.predict([[1,2]]))
print(predictor.predict([[1]]))
print(1)