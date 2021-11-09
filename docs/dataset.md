# Benchmark dataset

To evaluate the effectiveness of a prediction model on an arbitrary DNN model, we need a representative dataset that covers a large prediction scope. nn-Meter collects and generates 26k CNN models (Please refer the paper for the dataset generation method).

We release the dataset, and provide an interface of `nn_meter.dataset` for users to get access to the dataset. This interface could automatically download the nn-Meter bench dataset and return the path of the dataset when calling. Users can also download the data from the [Download Link](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip) on their own. This [example](../examples/nn-meter_predictor_for_bench_dataset.ipynb) shows how to use nn-Meter predictor to predict latency for the bench dataset.

**Note:** to measure the inference latency of models in this dataset, we generate tensorflow pb and tflite models and measure their latency on the target devices. However, since it requires hundreds of GB storage to store the full dataset, we didn't include these model files. Instead, we parse the pb files and record the model structures and parameters in 
`nn_meter.dataset`.

Since the dataset is encoded in a graph format, we also provide an interface of `nn_meter.dataset.gnn_dataloader` for GNN training. By this interface, `GNNDataset` and `GNNDataloader` build the model structure of the bench dataset in `.jsonl` format into GNN required dataset and data loader. Users could refer to this [example](../examples/nn-meter_dataset_for_gnn.ipynb) for further information of `gnn_dataloader`. Note that to apply nn-Meter bench dataset for GNN training, the package `torch` and `dgl` should be installed.

