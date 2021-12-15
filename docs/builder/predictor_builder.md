# usage

#### step1: sampling from prior distribution

Run the following scripts:

```
python init_sampler.py --config configs/conv.yaml
```

it takes input files from `configs`:  (i) kernel_type: `conv-bn-relu/dwconv-bn-relu,...etc`, (ii) inital number of data to sample: `conv-bn-relu`: 12000, `dwconv-bn-relu`: 5000, other kenels: 2000/4000

prior knowledge: we store it as csv files in `data_sampler/prior`

sampling code: `data_sampler/block_sampler.py->data_sampler/prior_distribution_sampler.py`

sampling stage: `prior`

kernel model generation: `generator/generator_block.py -> generator/networks`

**to-dos: if user has a new kernel, he/she needs to add: (i) configuration file in `configs`, (ii) tf kernel code generation in `generator/networks` (iii) collect the prior data and implement sampling code **

#### step2: measure latency on device

run the following scripts:

```
python measure_latency.py --savepath /path/to/the/savedmodels --outputfilepath /path/to/savedcsvfile
```

**to-dos (for us): implement code for our four backends (currently it connects to tflite cpu)**

#### step3: fine-grained sampling for data with large errors

run the following script:

```
python adaptive_sampler.py --kernel conv-bn-relu --sample_num 10 --iteration 10
```

In this stage, we conduct the following steps:

* build a regression model by current sampled data: `regression/build_regression_model.py, regression/extract_feature.py, regression/kernel_predictor.py`
* locate data points in testset with large prediction error (prediction error >`large_error_threshold, default=0.2`): `regression/build_regression_model.py`
* for each data point, we perform fine-grained data sampling:
  * `generator/generator_block.py, sampling_stage=finegrained'`
  * `data_sampler/block_sampler.py->data_sampler/finegrained_sampler.py`
  * `python measure_latency.py`
* collect sampled data, conduct next iteration
