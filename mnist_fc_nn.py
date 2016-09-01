# A fully connected neural network trained on MNIST images
#
# - Architecture:
#   784 neurons (input) -> 500 neurons -> 150 neurons -> 10 neurons (output)
import ml.datasets
from ml.nn import *

IMG_W = 28 # width of image
IMG_H = 28 # height of image

# define neural net
net = Net([
    WeightLayer.create_transition(IMG_W*IMG_H, 500),
    SigmoidLayer(),
    WeightLayer.create_transition(500, 150),
    SigmoidLayer(),
    WeightLayer.create_transition(150, 10),
    SigmoidLayer(),
    CrossEntropyLoss()
])

def process_dataset(X_imgs, Y_labels):
    X_ret = X_imgs.reshape((-1, IMG_W*IMG_H, 1)) - np.mean(X_imgs)
    X_ret /= np.max(X_imgs)

    Y_one_hot = np.zeros((len(Y_labels), 10, 1))
    for i in range(0, len(Y_labels)):
        digit = Y_labels[i]
        Y_one_hot[i] = np.eye(10)[digit].T

    return X_ret, Y_one_hot

train_in, train_out = ml.datasets.load_mnist("training")

plt.imshow(train_in[0])

print("loaded training set")
train_in, train_out = process_dataset(train_in, train_out)

# train using gradient descent for 600000 iterations
net.train_gd(train_in, train_out, 0.1, 200000)

train_accuracy = net.accuracy(train_in, train_out)
print("Training accuracy: %f" % train_accuracy)


# Test the trained net
test_in, test_out = ml.datasets.load_mnist("testing")
test_in, test_out = process_dataset(test_in, test_out)

test_accuracy = net.accuracy(test_in, test_out)
print("Test accuracy: %f" % test_accuracy)
