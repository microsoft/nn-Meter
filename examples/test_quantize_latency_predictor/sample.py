import random

blocks = 21
block_group_info = 5

def sample_active_subnet():
    ks_candidates = [3, 5, 7]
    expand_candidates = [3, 4, 6]
    depth_candidates = [2, 3, 4]

    # sample kernel size
    ks_setting = []
    ks_candidates = [ks_candidates for _ in range(blocks - 1)]
    for k_set in ks_candidates:
        k = random.choice(k_set)
        ks_setting.append(str(k))

    # sample expand ratio
    expand_setting = []
    expand_candidates = [expand_candidates for _ in range(blocks - 1)]
    for e_set in expand_candidates:
        e = random.choice(e_set)
        expand_setting.append(str(e))

    # sample depth
    depth_setting = []
    depth_candidates = [depth_candidates for _ in range(block_group_info)]
    for d_set in depth_candidates:
        d = random.choice(d_set)
        depth_setting.append(str(d))

    return "ks" + "".join(ks_setting) + \
        "_e" + "".join(expand_setting) + \
            "_d" + "".join(depth_setting)
