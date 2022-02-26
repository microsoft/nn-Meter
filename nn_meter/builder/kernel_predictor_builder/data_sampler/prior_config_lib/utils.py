import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_conv_zoo(filename = "conv.csv"):
    filename = os.path.join(BASE_DIR, filename)
    conv_df = pd.read_csv(filename)
    hws = conv_df["input_h"]
    cins = conv_df["cin"]
    couts = conv_df["cout"]
    ks = conv_df["ks"]
    strides = conv_df["stride"]
    return hws, cins, couts, ks, strides


def read_dwconv_zoo(filename = "dwconv.csv"):
    filename = os.path.join(BASE_DIR, filename)
    dwconv_df = pd.read_csv(filename)
    hws = dwconv_df["input_h"]
    cins = dwconv_df["cin"]
    ks = dwconv_df["ks"]
    strides = dwconv_df["stride"]
    return hws, cins, ks, strides


def read_fc_zoo(filename = "fc.csv"):
    filename = os.path.join(BASE_DIR, filename)
    fc_df = pd.read_csv(filename)
    cins = fc_df["cin"]
    couts = fc_df["cout"]
    return cins, couts


def read_pool_zoo(filename = "pooling.csv"):
    filename = os.path.join(BASE_DIR, filename)
    pool_df = pd.read_csv(filename)
    hws = pool_df["input_h"]
    cins = pool_df["cin"]
    ks = pool_df["ks"]
    strides = pool_df["stride"]
    return hws, cins, ks, strides
