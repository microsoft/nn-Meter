import random

def get_sampling_channels(c_start, c_end, c_ratio, c_layers):
    nc = []
    channel_scale = []
    while c_start <= c_end:
        channel_scale.append(c_start)
        c_start += c_ratio

    for _ in range(c_layers):
        index = random.choice(channel_scale)
        nc.append(index)
    return nc


def get_sampling_ks(kernelsizes, layers):
    return [random.choice(kernelsizes) for _ in range(layers)]


def get_sampling_es(es, layers):
    return [random.choice(es) for _ in range(layers)]
