# Examples for nn-Meter

In this folder, we provide several examples to show the usage of nn-Meter package.

The first example [1. Use nn-Meter for models with different format](nn-meter_for_different_model_format.ipynb) shows the basic python binding usage of nn-meter with models with different format of Tensorflow, PyTorch and ONNX model.

#### Benchmark dataset

To evaluate the effectiveness of a prediction model on an arbitrary DNN model, we need a representative dataset that covers a large prediction scope. nn-Meter collects and generates 26k CNN models. (Please refer the paper for the dataset generation method.)

We release the dataset, and provide an interface of `nn_meter.dataset` for users to get access to the dataset. Users can also download the data from the [Download Link](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip) on their own. 

Example [2. Use nn-Meter with the bench dataset](nn-meter_for_bench_dataset.ipynb) shows how to use nn-Meter to predict latency for the bench dataset.

Since the dataset is encoded in a graph format, we also provide an example [3. Use bench dataset for GNN training](gnn_for_bench_dataset.ipynb) of using GCN to predict the model latency with the bench dataset.

Finally, we provide more hardware-ware NAS examples in NNI.

## Examples list

1. [Use nn-Meter for models with different format](nn-meter_for_different_model_format.ipynb)
2. [Use nn-Meter with the bench dataset](nn-meter_for_bench_dataset.ipynb)
3. [Use bench dataset for GNN training](gnn_for_bench_dataset.ipynb)
4. Use nn-Meter to construct latency constraint in SPOS NAS (TBD)

   - [Use nn-Meter in search part](https://github.com/microsoft/nni/blob/master/examples/nas/oneshot/spos/multi_trial.py)
   - [Use nn-Meter in sampling part](https://github.com/microsoft/nni/blob/master/examples/nas/oneshot/spos/supernet.py)
5. [Use nn-Meter to construct latency penalty in Proxyless NAS](https://github.com/microsoft/nni/tree/master/examples/nas/oneshot/proxylessnas)
