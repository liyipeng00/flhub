# FLHub


This is a hub of Federated Learning and Split Learning.

## Datasets

MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100

## Models

MNIST and Fashion-MNIST
+ Logistic Regression
+ MLP
+ LeNet-5
+ CNN


| Model | Param |
| -- | -: |
| Logistic  | 7,850 |
| MLP | 50,890 |
| LeNet-5 | 44,426|
| CNN | 228,586 |


CIFAR-10, CIFAR-100, CINIC-10
+ Scaling VGG-9
+ ResNet-18 w/o / w/ group normalization

## Algorithms

### Federated Averaging (FedAvg)

+ main_fedavg.py   ($\checkmark$) <u>Less memory.</u> We aggregate the parameters after any client complete its local training to save the memory.
  $$
  \bar{x} = \sum_{s \in \mathcal{S}} \frac{1}{\sum_{i \in \mathcal{S}p_i}p_i} p_s x_s,
  $$
  where $x$ is model parameters, $\mathcal{S}\sim \mathcal{U}(M, S)$ is a subset uniformly sampled without replacement from $[M]$, and $p_s = n_s/n$ is the weight of client $s$.
  
+ main_fedavg2.py ($\times$)
  $$
  \bar{x} = \sum_{s \in \mathcal{S}} \frac{1}{\sum_{i \in \mathcal{S}p_i}p_i} p_s x_s,
  $$
  where $x$ is model parameters, $\mathcal{S}\sim \mathcal{U}(M, S)$ is a subset uniformly sampled without replacement from $[M]$, and $p_s = n_s/n$ is the weight of client $s$.

+ main_fedavg3.py ($\times$)
  $$
  \bar{x} = \sum_{s \in \mathcal{S}} \frac{1}{S} x_s,
  $$
  where $x$ is model parameters, $\mathcal{S}\sim \mathcal{W}(M, S, \mathbf{p})$ is a subset sampled with replacement from $[M]$ with probabilities $\{p_s\}$, and $p_s = n_s/n$ is the weight of client $s$.

+ main_fedavg4.py ($\times$)
  $$
  \bar{x} = \sum_{s \in \mathcal{S}} \frac{M}{S} p_s x_s,
  $$
  where $x$ is model parameters, $\mathcal{S}\sim \mathcal{U}(M, S)$ is a subset uniformly sampled without replacement from $[M]$, and $p_s = n_s/n$ is the weight of client $s$.









