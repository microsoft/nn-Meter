from nn_meter.dataset.generator.generate_model import ModelGenerator
generator = ModelGenerator("/data/jiahang/working/nn-Meter/nn_meter/dataset/generator/configs/vgg.yaml", "/data/jiahang/working/nn-Meter/examples/test_dataset_generator/")
generator.run()