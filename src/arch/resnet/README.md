# ResNet

## Dataset
CIFAR10 split into 40k training, 10k validation, and 10k testing. Downloaded from torchvision dataset library.

#### Problem
Increasing the number of layers in a neural network should theoretically increase performance since the model has a higher capacity to learn more complex representations.
However, deeper networks tend to perform worse or the accuracy saturates without overfitting. This is called degradation.

#### Approach
Instead of making the network learn the original desired mapping H(x) directly, it learns a residual mapping F(x), where:
F(x) = H(x) - x

#### Key Concepts


#### Data

