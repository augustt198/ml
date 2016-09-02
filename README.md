# ml

Machine learning framework written from scratch.

The goal is to keep the code simple and readable (the core library is ~300 lines) and written mostly
in Python (convolution is implemented in C to avoid a major bottleneck).

## Installing

- `git clone` the project somewhere
- `python setup.py install` to build C extensions and install
- `./setup_data` to download datasets (if you want to run the examples). This will download about 175 MB.

## Examples

- `mnist_fc_nn.py` trains a fully connected neural network on MNIST data. You should see ~96% accuracy.
- `mnist_cnn.py` trains a convnet on MNIST data. You should see 98% or higher accuracy.

## Usage

The `ml.nn` module is used to create and train neural networks.

The module contains these classes:

- `Net` – neural network composed of layers
  - **Constructor**
    - `Net(layer)` – creates a networks from an array of layers.
    The last layer should be a loss layer.
  - **Methods**
    - `forward_pass(input[, layers])` – feed `input` into the bottom layer and propagate
    forward, returning the result of the last layer. It propagates through `layers`,
    if given, or defaults to the layers given to the constructor.
    - `backward_pass(top_diff=1)` - perform backpropagation through the network, giving
    the top layer the gradient `top_diff` (1 if omitted). Returns the gradient of the
    input received during the forward pass. `forward_pass` must be called first!
    - `train_gd(X, Y, lr, iters)` – train the network on a dataset with inputs `X` and
    outputs `Y` via gradient descent with `iters` iterations and learning rate `lr`.
    - `accuracy(X, Y)` – test the accuracy of the network on a dataset with inputs `X` and
    outputs `Y`.

- `WeightLayer` – fully connected layer
  - **Constructor**
    - `WeightLayer(w, b)` - create a fully connected layer with weights `w` and biases `b` (both
    numpy arrays).
    - `WeightLayer.create_transition(units_bottom, units_top, init_type='random')` – creates a fully connected layer that transitions
    from a layer with `units_bottom` neurons to `units_top` neurons. `init_type` describes how the weights are initialized and can be
    either `'random'`, `'xavier'`, or `'xavier_relu'`
- `ConvLayer` - convolution layer
  - **Constructor**
    - `ConvLayer(in_depth, nfilters, fsize)` – creates a convolution layer that accepts a volume with depth `in_depth` and
    and outputs a volume with depth `nfilters` using a `fsize` x `fsize` sized kernel.
  - During the forward pass, a `ConvLayer` should be given a volume in the form of a numpy array with shape
  `(depth, height, width)`.
- `PoolLayer` - pooling layer
  - **Constructor**
    - `PoolLayer(block_size, pool_type='max')` – creates a pooling layer that downsamples the input volume
    using `block_size` x `block_size` blocks. Currently only max-pooling is implemented.
- `SigmoidLayer` - sigmoid activiation layer
  - **Constructor:** `SigmoidLayer()`
- `TanhLayer` - tanh activation layer
  - **Constructor:** `TanhLayer()`
- `ReLULayer` - ReLU (REctified Linear Unit) activation layer
  - **Constructor:** `ReLULayer()`
- `UnfurlLayer` - unfurls an arbitrarily dimensioned input into a column vector
  - **Constructor:** `UnfurlLayer()`
  - This layer is useful for feeding a convolution volume into a fully connected layer
- `SoftmaxLayer` - applies the softmax function so that outputs sum to one.
  - **Constructor:** `SoftmaxLayer()`
- `SquaredLoss` - squared loss between input and target
  - **Constructor:** `SquaredLoss()`
  - The target value must be set using the `target` attribute
- `CrossEntropyLoss` - cross entropy loss
  - **Constructor:** `CrossEntropyLoss()`
  - The target value must be set using the `target` attribute
- `DropoutLayer` - dropout layer (randomly kills neurons)
  - **Constructor**
    - `DropoutLayer(p)` – creates a dropout layer with probablity `p` of any given neuron dropping out
    (i.e. `p = 0.5` will kill about half of the input neurons).

All layer objects must have these two methods implemented:

- `forward(bottom)` – feeds `bottom` into the layer and returns the output.
- `backward(top_diff)` feeds the gradient `top_diff` into the layer and returns the bottom's gradient.
