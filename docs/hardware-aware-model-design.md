# Hardware-aware DNN Model Design

In many deep learning scenarios, such as model deployment in mobile devices, there are strict constraints on model inference efficiencies as well as the model accuracy. For example, the inference latency and energy consumption are the most frequently used criteria of efficiencies to determine whether a DNN (deep neural network) model could be deployed on a smart phone or not. A real-world DNN model designer has to consider the model efficiency. A typical methodology is to train a big-sized model to meet the accuracy requirements first, and then adopt algorithms like model compression, knowledge distillation, and neural architecture search (NAS), to get a light weight model with similar performance but much smaller size. 

However, as pointed out in our work [1] and many others, ***neither model storage size nor computations (measured by FLOPs) is a good indicator to measure the real inference efficiency (e.g., latency or energy consumption)***. Operators with similar FLOPs may have very different inference latency on different hardware platforms (e.g., CPU, GPU, and ASIC) (as shown in Figure 1 of [1]). This makes the effort of designing efficient DNN models for a target hardware bit of games of opening blind boxes.

## `nn-Meter` - towards transparency of the blind boxes

nn-Meter [2] is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices. The key idea of nn-Meter is dividing a whole model inference into kernels, i.e., the execution units on a device, and conducting kernel-level prediction.

To achieve the accurate latency prediction, nn-Meter provides the following abilities and artifacts

- kernel detection, which automatically detects the execution unit of model inference via a set of well-designed test cases; 
- model latency dataset, and a set of builtin predictors based on it. nn-Meter profiles a large dataset of 26,000 models on three popular platforms of edge hardware (mobile CPU, mobile GPU, and Intel VPU). The builtin predictors significantly outperform the prior state-of-the-art.
- adaptive sampling to efficiently sample the most beneficial configurations from a large space to build accurate kernel-level latency predictors. 

## Hardware-aware Neural Architecture Search

Compared with the conventional NAS algorithms, some recent works (i.e. hardware-aware NAS, aka HW-NAS) integrated hardware-awareness into the search loop and achieves a balanced trade-off between accuracy and hardware efficiencies [3]. 

As formulated in many works, the search space is one of the three key aspects of a NAS process (the other two are the search strategy and the evaluation methodology) and matters a lot to the final results. nn-Meter, with the latency predictor and the adaptive sampling techniques behind, could specialize the search space for the target hardware platform (shown as below).

![nw-nas](imgs/hw-nas.png)

Our HW-NAS framework [1] firstly automatically selects the hardware-friendly operators (or blocks) by considering both representation capacity and hardware efficiency. The selected operators could establish a ***hardware-aware search space*** for most of existing NAS algorithms. 

Besides the search space specialization, our HW-NAS framework also allows combining nn-Meter with existing NAS algorithms in the optimization objectives and constraints. 

## Hardware-aware Model Compression

## References
1. Zhang, Li Lyna, et al. ["Fast hardware-aware neural architecture search."]((https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Zhang_Fast_Hardware-Aware_Neural_Architecture_Search_CVPRW_2020_paper.pdf)) Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.
2. Zhang, Li Lyna, et al. ["nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices."](https://dl.acm.org/doi/10.1145/3458864.3467882) Proceedings of the 19th ACM International Conference on Mobile Systems, Applications, and Services (MobiSys 2021)
3. <div class="csl-entry">Benmeziane, H., Maghraoui, K. el, Ouarnoughi, H., Niar, S., Wistuba, M., &#38; Wang, N. (2021). <i>A Comprehensive Survey on Hardware-Aware Neural Architecture Search</i>. http://arxiv.org/abs/2101.09336</div>