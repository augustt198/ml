import numpy as np

import datasets
#np.set_printoptions(threshold=np.nan)

# sigmoid nonlinearity
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# returns the column vector representing
# the digit. E.g. 3 becomes [0,0,1,...,0]
def digit2vec(d):
    return np.eye(10)[:,d - 1]

# unrolls the pixels of an image to a column vector
def img2input(img):
    return img.reshape((img.shape[0]*img.shape[1], 1))

# Multilayer perceptron
class MLP:
    
    # `arch` -  the architecture of the neural net, specified
    #           as an array of numbers representing the number
    #           of units in each layer. For example: [400, 600, 10]
    #           describes 400 input units, one hidden layer with 600
    #           units, and 10 output units
    def __init__(self, arch):
        self.arch = arch

    def init_weights(self):
        self.weights = []
        for i in range(0, len(self.arch) - 1):
            units_current = self.arch[i]
            units_next    = self.arch[i + 1]
            #W = np.random.randn(units_next, units_current + 1) / np.sqrt(units_current)
            W = np.random.random((units_next, units_current + 1))
            W = W * 2 - 1 # -1 to 1 distribution
            self.weights.append(W)

    def forward_pass(self, in_layer):
        self.layers = [in_layer]
        for i in range(0, len(self.arch) -1):
            layer = self.layers[i]
            # add bias row
            layer = np.r_[np.ones((1,1)), layer]
            W     = self.weights[i]
            next_layer = W.dot(layer)
            next_layer = sigmoid(next_layer) # activation
            self.layers.append(next_layer)

        # return output layer (last layer)
        return self.layers[-1]

    # perform backprop with the expected value
    def backprop(self, expected):
        dJ_da = -(expected.reshape((expected.shape[0],1)) - self.layers[-1])
        weight_grads = []
        for i in range(len(self.arch) - 1, 0, -1):
            front_layer = self.layers[i]
            if i != len(self.arch) - 1:
                front_layer = np.r_[np.ones((1,1)), self.layers[i]]
            # add bias
            back_layer  = np.r_[np.ones((1,1)), self.layers[i - 1]]
            # sigmoid derivative
            da_dz = front_layer * (1 - front_layer)
            # chain rule
            dJ_dz = dJ_da * da_dz
            if i != len(self.arch) - 1:
                dJ_dz = dJ_dz[1:,:]
            
            dz_dW = back_layer
            dJ_dW = dJ_dz.dot(dz_dW.T)
            #print("!!!", dJ_dW)
            weight_grads = [dJ_dW] + weight_grads

            # gradients for previous layer's outputs
            dJ_da = self.weights[i - 1].T.dot(dJ_dz)

        return weight_grads

    # minibatch gradient descent
    def minibatch_gd(self, X, Y, iters=300000, batchsize=5):
        training_idx = 0
        m = X.shape[0]
        for i in range(0, iters):
            gradient_avg = None
            for j in range(0, batchsize):
                img = img2input(X[training_idx])
                self.forward_pass(img)
                W_grad = self.backprop(digit2vec(Y[training_idx]))
                if gradient_avg == None:
                    gradient_avg = W_grad
                else:
                    #gradient_avg += W_grad
                    gradient_avg = [x+y for x,y in zip(gradient_avg, W_grad)]

                # comment b/c fixed example
                training_idx = (training_idx + 1) % m
            for l in range(0, len(self.weights)):
                self.weights[l] -= 1.01 * (gradient_avg[l] / batchsize)
            #print(" ".join(map(str, map(np.linalg.norm, gradient_avg))))

    def grad_check(self, image, label):
        delta = 0.0001 # the h in (f(x+h) - f(x))/h

        output = self.forward_pass(image)
        cost_old = self.cost([output], [label])

        analytical_grad = self.backprop(label)
        numeric_grad = []
        for i in range(0, len(self.weights)):
            W = self.weights[i]
            W_grad = np.zeros(W.shape)
            for x in range(0, W.shape[0]):
                for y in range(0, W.shape[1]):
                    W_changed = W.copy()
                    W_changed[x,y] += delta
                    self.weights[i] = W_changed
                    output = self.forward_pass(image)
                    cost_new = self.cost([output], [label])
                    grad = (cost_new - cost_old) / delta
                    W_grad[x,y] = grad
            self.weights[i] = W # return to normal
            numeric_grad.append(W_grad)

        return numeric_grad, analytical_grad

                
    def cost(self, out, expected):
        s = 0
        for i in range(0, len(out)):
            s += np.sum(np.square(expected[i] - out[i]))
        return (0.5 / len(out)) * s

    def accuracy(self, imgs, labels):
        correct = 0
        for i in range(0, len(imgs)):
            in_layer = img2input(imgs[i])            
            out_layer = self.forward_pass(in_layer)
            if np.argmax(out_layer) + 1 == labels[i][0]:
                correct += 1
        return float(correct) / len(imgs)


X, Y = datasets.load_mnist("training")

# preprocess
for i in range(0, len(X)):
     X[i] /= 255.0

print("Preprocessing finished")

mlp = MLP([28*28, 30, 10])
mlp.init_weights()

#print("Before training accuracy: ", mlp.accuracy(X, Y))
out = mlp.forward_pass(img2input(X[0]))
expected = digit2vec(Y[0])

# expecteds = map(digit2vec, Y)
# results = map(lambda x: mlp.forward_pass(img2input(x)), X)
cost = mlp.cost([out], [expected])
print("Initial cost: %f" % (cost))
# 
# grad_backprop = mlp.backprop(expected)
# idx = np.unravel_index(grad_backprop[0].argmax(), grad_backprop[0].shape)
# print("max grad: ", grad_backprop[0][idx])
# delta = 0.001
# # for i in range(0, 28*28):
# #     print(i, img2input(X[0])[i])
# print(mlp.weights[0][idx])
# mlp.weights[0][idx] += delta
# changed_output = mlp.forward_pass(img2input(X[0]))
# changed_cost = mlp.cost([changed_output], [expected])
# grad_approx = (changed_cost - cost) / delta
# 
# print("Gradient approximation: ", grad_approx)

check = mlp.grad_check(img2input(X[0]), expected)
print("Numeric gradient", check[0][0])
print("Backprop gradient", check[1][0])

mlp.minibatch_gd(X, Y, 50000, 1)
out = mlp.forward_pass(img2input(X[0]))
cost = mlp.cost([out], [expected])
print(mlp.backprop(expected))
# 

expecteds = map(digit2vec, Y)
results = map(lambda x: mlp.forward_pass(img2input(x)), X)
cost = mlp.cost([out], [expected])
print("output", out)
print("After training cost: %f" % (cost))
print(mlp.accuracy(X, Y))

