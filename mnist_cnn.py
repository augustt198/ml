# Convolutional Neural Network trained on MNIST images
#
# - network architecture is similar to LeNet-5
# - if an argument is given, the trained Net object
#   will be saved to that path

import ml.datasets
from ml.nn import *

IMG_W = 28 # width of image
IMG_H = 28 # height of image

# define neural net
net = Net([
    ConvLayer(in_depth=1, nfilters=6, fsize=5), # -> 6x24x24
    ReLULayer(),
    PoolLayer(block_size=2), # -> 6x12x12
    ConvLayer(in_depth=6, nfilters=16, fsize=3), # -> 16x10x10
    ReLULayer(),
    PoolLayer(block_size=2), # -> 16x5x5
    UnfurlLayer(), # -> 16*5*5
    WeightLayer.create_transition(16*5*5, 120, init_type='xavier_relu'),
    TanhLayer(),
    WeightLayer.create_transition(120, 84, init_type='xavier_relu'),
    TanhLayer(),
    WeightLayer.create_transition(84, 10, init_type='xavier_relu'),
    TanhLayer(),
    SoftmaxLayer(),
    CrossEntropyLoss()
])

def process_dataset(X_imgs, Y_labels):
    X_ret = X_imgs.reshape((-1, 1, IMG_W, IMG_H)) - np.mean(X_imgs)
    X_ret /= np.max(X_imgs)

    Y_one_hot = np.zeros((len(Y_labels), 10, 1))
    for i in range(0, len(Y_labels)):
        digit = Y_labels[i]
        Y_one_hot[i] = np.eye(10)[digit].T

    return X_ret, Y_one_hot

train_in, train_out = ml.datasets.load_mnist("training")
train_in, train_out = process_dataset(train_in, train_out)

train_accuracy = net.accuracy(train_in, train_out)
print("Initial training accuracy: %f" % train_accuracy)

# train using gradient descent for 3 epochs
net.train_gd(train_in, train_out, 0.01, 60000 * 3)

train_accuracy = net.accuracy(train_in, train_out)
print("Final Training accuracy: %f" % train_accuracy)


# Test the trained net
test_in, test_out = ml.datasets.load_mnist("testing")
test_in, test_out = process_dataset(test_in, test_out)

test_accuracy = net.accuracy(test_in, test_out)
print("Test accuracy: %f" % test_accuracy)

# Save net
if len(sys.argv) > 1:
    import pickle
    import sys
    print("Saving network")
    with open(sys.argv[1], 'wb') as f:
        pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
