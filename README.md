# CE-FEDL
**Communication Efficient (CE) - Federated Learning (FEDL)**

This project aims to compare different methods of improving the communication efficiency of federated learning. The goal is to reduce the communication overhead, while still retaining good performance.

For the experiments, the [CIFAR-10 and FEMNIST datasets](data.py) are used with a [500k parameter convolutional model](models.py).

## Data and Models

To use the datasets and models, first download the FEMNIST dataset - link is in [data.py](data.py). Then run:
```python
import data
import models

femnist_model = models.create_model("femnist")
femnist_trainloaders, femnist_testloaders = data.femnist_data() 

cifar_model = models.create_model("cifar")
cifar_trainloaders, cifar_testloaders = data.cifar_data() 
```
## Methods 

The three methods being compared are:
- [Federated Distillation](fed_dist)
- [Aggregation](aggregation_methods)
- [Sparsification](sparsification_method_natasha)
