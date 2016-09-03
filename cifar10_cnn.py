import sys
sys.path.pop(0) # hax

import ml.datasets
import ml.preprocess
from ml.nn import *

IMG_W, IMG_H = 32, 32

net = Net([
    ConvLayer(in_depth=3, nfilters=64, fsize=5),                    # -> 64x28x28
    ReLULayer(),                                                    # -> 64x28x28
    PoolLayer(block_size=2),                                        # -> 64x14x14
    ConvLayer(in_depth=64, nfilters=64, fsize=5),                   # -> 64x10x10
    ReLULayer(),                                                    # -> 64x10x10
    PoolLayer(block_size=2),                                        # -> 64x5x5
    UnfurlLayer(),                                                  # -> 1600
    WeightLayer.create_transition(1600, 200, init_type='xavier'),   # -> 200
    TanhLayer(),                                                    # -> 200
    WeightLayer.create_transition(200, 30, init_type='xavier'),     # -> 30
    TanhLayer(),
    WeightLayer.create_transition(30, 10, init_type='xavier'),      # -> 10
    TanhLayer(),
    SoftmaxLayer(),
    CrossEntropyLoss()
])

train_in, train_out = ml.datasets.load_cifar10_training()
train_in = ml.preprocess.rescale(train_in)

epochs = 6
print("Training for %d epochs" % epochs)

train_accuracy = net.accuracy(train_in, train_out)
print("Initial training accuracy: %f" % train_accuracy)

net.train_gd(train_in, train_out, 0.2, len(train_in) * epochs)

train_accuracy = net.accuracy(train_in, train_out)
print("Final training accuracy: %f" % train_accuracy)

test_in, test_out = ml.datasets.load_cifar10_test()
test_in = ml.preprocess.rescale(test_in)

test_accuracy = net.accuracy(test_in, test_out)
print("Test accuracy: %f" % test_accuracy)
