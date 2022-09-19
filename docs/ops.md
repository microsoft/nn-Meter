# Supported Operators

nn-Meter currently supports the major operators in CNN models. The following table lists out the tested operators:

| Type        | ops                                 |
| :---------- | :---------------------------------- |
| Conv        | conv, dwconv                        |
| Activations | relu, relu6, sigmoid, swish, hswish |
| Pooling     | avgpool, maxpool, global_avg_pool   |
| FC          | fc                                  |
| BN          | bn                                  |
| others      | se, channelshuffle, add, concat     |

According to our knowledge, the listed operators have already covered the major frequently used ones in CNN models. If there are new operators in your model, please read the followings to decide whether nn-Meter can accurately predict the latency.

## Operators' latency vs final model latency

Due to the limited resources on edge and mobile devices, the latency of Conv related operators dominates the model latency. Therefore, at most cases, the accurate prediction of Conv operators indicate that the predicted model latency is also accurate. 

However, the predicted model latency may be less accurate on the AI accelerators, since the memory access cost is relatively high, and element-wise operators can take large latency percentages.

## New operators (e.g., self-attention)

* **New inventing operators:** If the new inventing operators are computation-intensive, then you need build the latency predictors for them. (We will opensource the algorithm for building latency predictors for new operators later)
  Otherwise, the new inventing operators are memory-intensive, you can ignore it for mobile CPUs and mobile GPUs. It's better to build the latency predictors for them on AI accelerators.
* **NLP operators:** The current code has no implementation for the Transformer-based NLP models, and hence we don't have the latency predictors for new operators like self-attention. We plan to support in later versions.
