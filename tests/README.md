In nn-Meter/tests, we implement the integration test for all usages of nn-Meter. 

## Integration test

According to [nn-Meter usage](nn-Meter/docs/usage.md), nn-Meter is a latency predictor of models with type of Tensorflow, PyTorch, Onnx, nn-meter IR graph and NNI IR graph. In integration test, we run the test for mentioned models, collect the latency results, and compare the results with the reference results. For time saving and readability, we separate the integration test into two scripts with PyTorch model and others, respectively. 

For PyTorch model, we accomplished two graph converters, namely NNI-based torch converter and ONNX-based torch converter (Refer to [this doc](docs/usage.md#torch-model-converters) for more information). We test both converters in `tests/integration_test_torch.py`. Note that the NNI-based torch converter needs API from `nni.retiarii.nn.pytorch` (view [NNI doc](https://nni.readthedocs.io/en/stable/NAS/QuickStart.html#define-base-model)) to build the torch module, thus we collected torchvision models and modified the import package to meet NNI requirements. The modified model are saved in tests/torchmodels.


## github actions workflow

[GitHub Actions](https://docs.github.com/en/actions) workflow can automatically run the testing scripts along with a  PUSH action happens. Here we built three integration test yml scripts in nn-Meter/.github/workflows. Regarding the running time of testing for NNI-based torch and ONNX-based torch is long, we split the two test into two scripts file so that the tests can parallel run.