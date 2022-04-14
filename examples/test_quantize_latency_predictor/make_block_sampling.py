from .nas_models.common import make_divisible
import numpy as np

width_mults = [0.65, 0.8, 1.0, 1.2]
hw_candidate = [3, 5, 6, 7, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 40, 44,
                48, 52, 56, 80, 88, 96, 104, 112, 160, 176, 192, 208, 224]
strides_candidate = [1, 2] # suggest use prior distribution
ks_candidate = [3, 5, 7] # suggest use prior distribution
expand_ratio = [3, 4, 5, 6]
base_stage_width = [3, 16, 16, 24, 40, 80, 112, 160, 960, 1280]

cin_cout_candidate = []
for width_mult in width_mults:
    cin_cout_candidate.extend([make_divisible(w * width_mult) for w in base_stage_width])
print(list(np.unique(cin_cout_candidate)))

print()
