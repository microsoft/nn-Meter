from tensorflow import keras
from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, mhsa_with_multi_head_position_windows_layer_norm





for channel in [64, 72, 80, 96, 108, 112, 128]:
    model_input = keras.Input(shape=[196, channel], batch_size=1)
    
    # # nasvit
    # mark = 'nasvit'
    # v_scale = 4
    # k_dim = 8
    # ACT = 'swish'
    # model_output = mhsa_with_multi_head_position_windows_layer_norm(model_input, channel, channel // k_dim, k_dim, k_dim * v_scale, 1, nasvit_arch=True, activation=ACT, name="123")

    # ours
    mark = 'ours'
    v_scale = 4 # 1-6
    ACT = 'hard_swish'
    k_dim = 16
    model_output = mhsa_with_multi_head_position_windows(model_input, channel, channel // k_dim, k_dim, k_dim * v_scale, 1, nasvit_arch=False, activation=ACT, name="123")


    model = keras.Model(model_input, model_output)
    print(model_output.shape)


    model_path = f"/data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/{mark}_vs{v_scale}_k{k_dim}_c{channel}"
    model.save(model_path)

    # convert model to tflite model
    import sys, shutil
    sys.path.append("/data/data0/jiahang/nn-Meter/examples/test_transformer/implementation/nasvit")
    from nasvit_tf import tf2tflite
    tf2tflite(model_path, model_path + ".tflite")
    shutil.rmtree(model_path)