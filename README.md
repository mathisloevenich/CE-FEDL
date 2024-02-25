# CE-FEDL
**Communication Efficient (CE) - Federated Learning (FEDL)**

This project aims to compare different methods of improving the communication efficiency of federated learning. The goal is to reduce the communication overhead, while still retaining good performance.

For the experiments, the [CIFAR-10 and FEMNIST datasets](data.py) are used with a [500k parameter convolutional model](models.py).

## Methods 

The three methods being compared are:
- [Federated Distillation](fed_dist_flower)
- [One-shot](aggregation_methods)
- [Sparsification](sparsification)
