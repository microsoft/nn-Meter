# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import random
import torch
import jsonlines
from .bench_dataset import bench_dataset
from nn_meter.utils import get_user_data_folder
from nn_meter.utils.import_package import try_import_dgl


RAW_DATA_URL = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip"
__user_dataset_folder__ = os.path.join(get_user_data_folder(), 'dataset')


hws = [
    "cortexA76cpu_tflite21",
    "adreno640gpu_tflite21",
    "adreno630gpu_tflite21",
    "myriadvpu_openvino2019r2",
]


class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, device="cortexA76cpu_tflite21", split_ratio=0.8):
        """
        Dataloader of the Latency Dataset

        Parameters
        ----------
        data_dir : string
            Path to save the downloaded dataset
        train: bool
            Get the train dataset or the test dataset
        device: string
            The Device type of the corresponding latency
        shuffle: bool
            If shuffle the dataset at the begining of an epoch
        batch_size: int
            Batch size.
        split_ratio: float
            The ratio to split the train dataset and the test dataset.
        """
        err_str = "Not supported device type"
        assert device in hws, err_str
        self.device = device
        self.train = train
        self.split_ratio = split_ratio
        self.adjs = {}
        self.attrs = {}
        self.nodename2id = {}
        self.id2nodename = {}
        self.op_types = set()
        self.opname2id = {}
        self.raw_data = {}
        self.name_list = []
        self.latencies = {}
        self.data_dir = bench_dataset(data_folder=__user_dataset_folder__)
        self.load_model_archs_and_latencies(self.data_dir)
        self.construct_attrs()
        self.name_list = list(filter(lambda x: x in self.latencies, self.name_list))

    def load_model_archs_and_latencies(self, data_dir):
        for filename in data_dir:
            self.load_model(filename)

    def load_model(self, fpath):
        """
        Load a concrete model type.
        """
        assert os.path.exists(fpath), '{} does not exists'.format(fpath)

        with jsonlines.open(fpath) as reader:
            _names = []
            for obj in reader:
                if obj[self.device]:
                    _names.append(obj['id'])
                    self.latencies[obj['id']] = float(obj[self.device])

            _names = sorted(_names)
            split_ratio = self.split_ratio if self.train else 1-self.split_ratio
            count = int(len(_names) * split_ratio)

            if self.train:
                _model_names = _names[:count]
            else:
                _model_names = _names[-1*count:]

            self.name_list.extend(_model_names)

        with jsonlines.open(fpath) as reader:
            for obj in reader:
                if obj['id'] in _model_names:
                    model_name = obj['id']
                    model_data = obj['graph']
                    self.parse_model(model_name, model_data)
                    self.raw_data[model_name] = model_data
    
    def construct_attrs(self):
        """
        Construct the attributes matrix for each model.
        Attributes tensor:
        one-hot encoded type + input_channel , output_channel,
        input_h, input_w + kernel_size + stride
        """
        op_types_list = list(sorted(self.op_types))
        for i, _op in enumerate(op_types_list):
            self.opname2id[_op] = i
        n_op_type = len(self.op_types)
        attr_len = n_op_type + 6
        for model_name in self.raw_data:
            n_node = len(self.raw_data[model_name])
            # print("Model: ", model_name, " Number of Nodes: ", n_node)
            t_attr = torch.zeros(n_node, attr_len)
            for node in self.raw_data[model_name]:
                node_attr = self.raw_data[model_name][node]
                nid = self.nodename2id[model_name][node]
                op_type = node_attr['attr']['type']
                op_id = self.opname2id[op_type]
                t_attr[nid][op_id] = 1
                other_attrs = self.parse_node(model_name, node)
                t_attr[nid][-6:] = other_attrs
            self.attrs[model_name] = t_attr

    def parse_node(self, model_name, node_name):
        """
        Parse the attributes of specified node
        Get the input_c, output_c, input_h, input_w, kernel_size, stride
        of this node. Note: filled with 0 by default if this doesn't have
        coressponding attribute.
        """
        node_data = self.raw_data[model_name][node_name]
        t_attr = torch.zeros(6)
        op_type = node_data['attr']['type']
        if op_type =='Conv2D':
            weight_shape = node_data['attr']['attr']['weight_shape']
            kernel_size, _, in_c, out_c = weight_shape
            stride, _= node_data['attr']['attr']['strides']
            _, h, w, _ = node_data['attr']['output_shape'][0]
            t_attr = torch.tensor([in_c, out_c, h, w, kernel_size, stride])
        elif op_type == 'DepthwiseConv2dNative':
            weight_shape = node_data['attr']['attr']['weight_shape']
            kernel_size, _, in_c, out_c = weight_shape
            stride, _= node_data['attr']['attr']['strides']
            _, h, w, _ = node_data['attr']['output_shape'][0]
            t_attr = torch.tensor([in_c, out_c, h, w, kernel_size, stride])
        elif op_type == 'MatMul':
            in_node = node_data['inbounds'][0]
            in_shape = self.raw_data[model_name][in_node]['attr']['output_shape'][0]
            in_c = in_shape[-1]
            out_c = node_data['attr']['output_shape'][0][-1]
            t_attr[0] = in_c
            t_attr[1] = out_c
        elif len(node_data['inbounds']):
            in_node = node_data['inbounds'][0]
            h, w, in_c, out_c = 0, 0, 0, 0
            in_shape = self.raw_data[model_name][in_node]['attr']['output_shape'][0]
            in_c = in_shape[-1]
            if 'ConCat' in op_type:
                for i in range(1, len(node_data['in_bounds'])):
                    in_shape = self.raw_data[node_data['in_bounds']
                                             [i]]['attr']['output_shape'][0]
                    in_c += in_shape[-1]
            if len(node_data['attr']['output_shape']):
                out_shape = node_data['attr']['output_shape'][0]
                # N, H, W, C
                out_c = out_shape[-1]
                if len(out_shape) == 4:
                    h, w = out_shape[1], out_shape[2]
            t_attr[-6:-2] = torch.tensor([in_c, out_c, h, w])

        return t_attr

    def parse_model(self, model_name, model_data):
        """
        Parse the model data and build the adjacent matrixes
        """
        n_nodes = len(model_data)
        m_adj = torch.zeros(n_nodes, n_nodes, dtype=torch.int32)
        id2name = {}
        name2id = {}
        tmp_node_id = 0
        # build the mapping between the node name and node id

        for node_name in model_data.keys():
            id2name[tmp_node_id] = node_name
            name2id[node_name] = tmp_node_id
            op_type = model_data[node_name]['attr']['type']
            self.op_types.add(op_type)
            tmp_node_id += 1

        for node_name in model_data:
            cur_id = name2id[node_name]
            for node in model_data[node_name]['inbounds']:
                if node not in name2id:
                    # weight node
                    continue
                in_id = name2id[node]
                m_adj[in_id][cur_id] = 1
            for node in model_data[node_name]['outbounds']:
                if node not in name2id:
                    # weight node
                    continue
                out_id = name2id[node]
                m_adj[cur_id][out_id] = 1
        
        for idx in range(n_nodes):
            m_adj[idx][idx] = 1

        self.adjs[model_name] = m_adj
        self.nodename2id[model_name] = name2id
        self.id2nodename[model_name] = id2name

    def __getitem__(self, index):
        model_name = self.name_list[index]
        return (self.adjs[model_name], self.attrs[model_name]), self.latencies[model_name], self.op_types

    def __len__(self):
        return len(self.name_list)


class GNNDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, shuffle=False, batchsize=1):
        self.dataset = dataset
        self.op_num = len(dataset.op_types)
        self.shuffle = shuffle
        self.batchsize = batchsize
        self.length = len(self.dataset)
        self.indexes = list(range(self.length))
        self.pos = 0
        self.graphs = {}
        self.latencies = {}
        self.construct_graphs()

    def construct_graphs(self):
        dgl = try_import_dgl()
        for gid in range(self.length):
            (adj, attrs), latency, op_types = self.dataset[gid]
            u, v = torch.nonzero(adj, as_tuple=True)
            graph = dgl.graph((u, v))
            MAX_NORM = torch.tensor([1]*len(op_types) + [6963, 6963, 224, 224, 11, 4])
            attrs = attrs / MAX_NORM
            graph.ndata['h'] = attrs
            self.graphs[gid] = graph
            self.latencies[gid] = latency

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self.pos = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        dgl = try_import_dgl()
        start = self.pos
        end = min(start + self.batchsize, self.length)
        self.pos = end
        if end - start <= 0:
            raise StopIteration
        batch_indexes = self.indexes[start:end]
        batch_graphs = [self.graphs[i] for i in batch_indexes]
        batch_latencies = [self.latencies[i] for i in batch_indexes]
        return torch.tensor(batch_latencies), dgl.batch(batch_graphs)
