import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_conv_zoo(filename = "conv.csv"):
    filename = os.path.join(BASE_DIR, filename)
    f = open(filename,'r')
    i = 0
    hws, cins, couts, ks, groups, strides = [], [], [], [], [], []   
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            # model, input_h, input_w, cin, cout, ks, stride, groups
            content = line.strip().split(',')
            hws.append(int(content[1]))
            cins.append(int(content[3]))
            couts.append(int(content[4]))
            ks.append(int(content[5]))
            strides.append(int(content[6]))
            groups.append(int(content[7]))
        i += 1
    return hws, cins, couts, ks, groups, strides


def read_dwconv_zoo(filename = "dwconv.csv"):
    filename = os.path.join(BASE_DIR, filename)
    f = open(filename,'r')
    i = 0
    hws, cins, ks, strides = [], [], [], []
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            # model, input_h, input_w, cin, cout, ks, stride, groups
            content = line.strip().split(',')
            hws.append(int(content[1]))
            cins.append(int(content[3]))
            ks.append(int(content[5]))
            strides.append(int(content[6]))
        i += 1
    return hws, cins, ks, strides


def read_fc_zoo(filename = "fc.csv"):
    filename = os.path.join(BASE_DIR, filename)
    f = open(filename,'r')
    cins, couts = [], []
    i = 0
    while True:
        line = f.readline()
        if not line:
            break
        if i > 0:
            # model, cin, cout
            content = line.strip().split(',')
            cins.append(int(content[1]))
            couts.append(int(content[2]))
        i += 1
    return cins, couts


def read_pool_zoo(filename = "poolings.csv"):
    filename = os.path.join(BASE_DIR, filename)
    cins, hws, ks, strides = [], [], [], []
    i = 0 
    f = open(filename,'r')
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            # model, input_h, input_w, cin, cout, ks, stride
            content = line.strip().split(',')
            hws.append(int(content[1]))
            cins.append(int(content[3]))
            ks.append(int(content[5]))
            strides.append(int(content[6]))
        i += 1 
    return hws, cins, ks, strides
