# Nasbench201 TensorFlow & TFLite Dump Tool
## Usage:
```bash
python3 nasbench201_generator.py -i nasbench201_acc.json -o nasbench201_pb -f nasbench201_tflite -t 24
```
## Args:
 - **-i** Input json string. Format:
  ```json
  {
    "1":{
        "config":{
            "name":"infer.tiny", 
            "C":16, 
            "N":5, 
            "arch_str":"|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|", 
            "num_classes":10
        }, 
        "acc":93.66000000000001
    }, 
    "5":{
        "config":{
            "name":"infer.tiny", 
            "C":16, 
            "N":5, 
            "arch_str":"|nor_conv_1x1~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_3x3~0|none~1|avg_pool_3x3~2|", 
            "num_classes":10
        }, 
        "acc":91.08666666666666
    }, 
    "6":{
        "config":{
            "name":"infer.tiny", 
            "C":16, 
            "N":5, 
            "arch_str":"|nor_conv_3x3~0|+|none~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|", 
            "num_classes":10
        }, 
        "acc":92.50666666666666
    }
}
  ```
  - **-o** Output frozen pb dir
  - **-f** Output tflite dir
  - **-t** Number of threads running the conversion

## Requirements:
### Python packages
```
enum34 == 1.1.10
h5py == 2.10.0
Keras-Applications == 1.0.8
Keras-Preprocessing == 1.1.0
numpy == 1.16.6
pathlib2 == 2.3.5
protobuf == 3.11.3
six == 1.14.0
tensorboard == 1.14.0
tensorflow == 1.14.0
tensorflow-estimator == 1.14.0
tqdm
```