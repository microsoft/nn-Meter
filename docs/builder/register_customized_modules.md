# Register customized algorithms as builtin modules

## Overview

## Register customized modules as builtin modules

We provide five types of modules, backends, kernels, 

### Step 1: create a customized backend/predictor/test case/operator/kernel

### Step 2: Prepare meta file

Create a yaml file with following keys as meta file:

`moduleType`: type of algorithms, could be one of tuner, assessor, advisor

`registerName`: builtin name used in experiment configuration file

`className`: tuner class name, including its module name, for example: demo_tuner.DemoTuner

`packageLocation`: /data/jiahang/working/tftest/test_package_import

`moduleFeatures`: class args validator class name, including its module name, for example: demo_tuner.MyClassArgsValidator


Following is an example of the yaml file:

```yaml
algoType: tuner
builtinName: demotuner
className: demo_tuner.DemoTuner
classArgsValidator: demo_tuner.MyClassArgsValidator
```

### Step 3: Register customized modules into nn-Meter

### Step 4: Test the registered module

## use the installed customized modules in experiment

# Manage builtin moduls using `nn-meter module`

## List builtin algorithms