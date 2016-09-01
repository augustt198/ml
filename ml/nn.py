import numpy as np

class WeightLayer:

    @staticmethod
    def create_transition(units_bottom, units_top, init_type='random'):
        if init_type == 'random':
            w = np.random.randn(units_bottom, units_top)
        elif init_type == 'xavier':
            w = np.random.randn(units_bottom, units_top) / np.sqrt(units_bottom)
        elif init_type == 'xavier_relu':
            w = np.random.randn(units_bottom, units_top) / np.sqrt(units_bottom / 2)
        else:
            raise "Unknown initialization type"
        bias = np.zeros((units_top, 1))
        return WeightLayer(w, bias)

    # Initial with weights w and bias b
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.values = None

    # f = wx + b
    def forward(self, bottom):
        self.values = np.dot(self.w.T, bottom) + self.b
        self.inputs = bottom

        return self.values

    def backward(self, top_diff):
        self.dJ_dw = np.dot(self.inputs, top_diff.T)
        self.dJ_db = top_diff * np.ones(self.b.shape)
        # gradients for previous layer's outputs
        dJ_da = np.dot(self.w, top_diff)

        return dJ_da

    def update_params(self, lr):
        self.w -= lr * self.dJ_dw
        self.b -= lr * self.dJ_db

class SigmoidLayer:

    # sigmoid nonlinearity: squashes input into
    # (0, 1) range
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # computes the sigmoid of the layer
    def forward(self, bottom):
        self.value = SigmoidLayer.sigmoid(bottom)
        return self.value

    # computes d/dz of the sigmoid = sigmoid(z) (1 - sigmoid(z))
    def backward(self, top_diff):
        df_dz = self.value * (1 - self.value)
        return top_diff * df_dz

class ReLULayer:
    def forward(self, bottom):
        mask = bottom < 0
        self.value = bottom.copy()
        self.value[mask] = 0
        return self.value

    def backward(self, top_diff):
        grad = self.value.copy()
        grad[self.value <= 0] = 0.1
        grad[self.value > 0]  = 1
        return top_diff * grad

class TanhLayer:
    def forward(self, bottom):
        self.value = np.tanh(bottom)
        return self.value

    def backward(self, top_diff):
        # d/dx tanh x = 1 - tanh^2 x
        grad = 1 - self.value**2
        return top_diff * grad

class SoftmaxLayer:
    
    # compute softmax:
    # h_i = exp(z_1) / sum(exp(z))
    def forward(self, bottom):
        # numerically stable
        bottom = bottom - np.max(bottom)

        raised = np.exp(bottom)
        self.values = raised / np.sum(raised)
        return self.values

    # derivative of softmax:
    # dh_i/dz_j = h_i(1 - h_j) if i = j
    # dh_i/dz_j = h_i(0 - h_j) if i != j
    def backward(self, top_diff):
        dim = self.values.shape[0]
        jacobian = np.eye(dim)
        jacobian -= np.tile(self.values.T, (dim, 1))
        jacobian *= np.tile(self.values, (1, dim))
        res = np.dot(jacobian, top_diff)
        return res

class DropoutLayer:
    # initializes a dropout layer with probability
    # p of neurons dropping out
    def __init__(self, p):
        self.p = p

    def forward(self, bottom):
        self.mask = np.random.random(bottom.shape) < self.p
        bottom = bottom.copy()
        bottom[self.mask] = 0
        return bottom

    def backward(self, top_diff):
        top_diff[self.mask] = 0
        return top_diff
        

class SquaredLoss:
    def __init__(self):
        self.target = None

    # squared loss: J = 0.5(h - y)^2
    def forward(self, bottom):
        self.inputs = bottom
        return 0.5 * np.sum((bottom - self.target) ** 2)

    # computes dJ/dh = h - y
    def backward(self, top_diff):
        return top_diff * (self.inputs - self.target)

class ConvLayer:
    def __init__(self, in_depth, nfilters, fsize, padding=0):
        self.in_depth = in_depth
        self.nfilters = nfilters
        self.fsize = fsize
        self.padding = padding

        self.init_parameters()

    def init_parameters(self):
        self.weights = np.random.random((self.nfilters, self.in_depth, self.fsize, self.fsize))
        self.weights /= self.weights.size
        self.biases = np.random.random((self.nfilters)) / self.nfilters

    # - `V` is the 3D input volume, indexed by V[channel, y, x]
    # - `K` is the 4D kernel stack, indexed by K[out_channel, in_channel, y, x]
    # - `b` (optional) is a 1D array of biases for each kernel, whose length is
    #       equal to K's first dimension length
    #
    # returns the result of convolving V with K (V * K = Z)
    @staticmethod
    def convolution(V, K, b=None):
        V_depth, V_h, V_w = V.shape
        chns_out, chns_in, filter_h, filter_w = K.shape

        Z_h = V_h - filter_h + 1
        Z_w = V_w - filter_w + 1
        Z = np.zeros((chns_out, Z_h, Z_w))

        for n in range(0, chns_out):
            kernel = K[n]
            bias = b[n] if b is not None else 0
            for y in range(0, V_h - filter_h + 1):
                for x in range(0, V_w - filter_w + 1):
                    extent = V[:, y:y+filter_h, x:x+filter_w]
                    val = np.sum(extent * kernel) + bias

                    Z[n, y, x] = val
        
        return Z

    # computes a convolution on the bottom volume
    def forward(self, bottom):
        p = self.padding

        padded = np.pad(bottom, pad_width=((0,0), (p,p), (p,p)), mode='constant', constant_values=0)

        self.cached_bottom = padded

        return ConvLayer.convolution(padded, self.weights, self.biases)

    def backward(self, top_diff):
        top_diff_fsize = top_diff.shape[1]

        self.bias_grad = np.sum(top_diff, (1, 2))

        self.weights_grad = np.zeros(self.weights.shape)

        for i, j in np.ndindex(self.weights.shape[:2]):
            in_map = self.cached_bottom[[j]]
            diff_map = np.array([ top_diff[[i]] ])
            k_slice = ConvLayer.convolution(in_map, diff_map)
            self.weights_grad[i,j] = k_slice[0]

        pad_amt_h = self.weights.shape[2] - 1
        pad_amt_w = self.weights.shape[3] - 1
        padded_top = np.pad(top_diff,
            pad_width=((0,0), (pad_amt_h,pad_amt_h), (pad_amt_w,pad_amt_w)),
            mode='constant', constant_values=0)

        flipped_weights = self.weights[:, :, ::-1, ::-1]
        # reverse channel modification
        flipped_weights = flipped_weights.swapaxes(0, 1)
        input_grad = ConvLayer.convolution(padded_top, flipped_weights)

        return input_grad

    def update_params(self, lr):
        self.weights -= lr * self.weights_grad
        self.biases -= lr * self.bias_grad

class PoolLayer:
    def __init__(self, block_size, pool_type='max'):
        self.block_size = block_size
        if pool_type != 'max':
            raise "Unsupported pooling type"
        self.pool_type = pool_type

    def forward(self, bottom):
        self.input = bottom
        bs = self.block_size

        chns, h, w = bottom.shape

        self.grad = np.zeros(bottom.shape)

        y_max_pool = bottom.reshape(chns, h / bs, bs, w).max(axis=2)
        max_pool = y_max_pool.reshape(chns, h / bs, w / bs, bs).max(axis=3) 

        max_pool_upscaled = max_pool.repeat(bs, axis=1).repeat(bs, axis=2)
        self.grad = np.equal(bottom, max_pool_upscaled).astype('int')

        return max_pool

    def backward(self, top_diff):
        upsampled_diff = top_diff.repeat(self.block_size, axis=1) \
            .repeat(self.block_size, axis=2)
        grad = upsampled_diff * self.grad
        return grad

class UnfurlLayer:
    def forward(self, bottom):
        self.original_shape = bottom.shape
        return bottom.reshape((bottom.size, 1))

    def backward(self, top_diff):
        return top_diff.reshape(self.original_shape)

class CrossEntropyLoss:
    
    # cross entropy loss:
    # J = -y*log(h) -(1 - y)*log(1 - h)
    def forward(self, bottom):
        self.inputs = bottom
        J = -self.target*np.log(bottom) - (1 - self.target)*np.log(1 - bottom)
        return np.sum(J)

    # dJ/dh = - (y - h)/(h(1 - h))
    def backward(self, top_diff):
        return -top_diff * (self.target - self.inputs) / (self.inputs * (1 - self.inputs))

class Net:

    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, input, layers = None):
        if layers == None:
            layers = self.layers
        current_vals = input
        for layer in layers:
            current_vals = layer.forward(current_vals)

        return current_vals

    def backward_pass(self, top_diff = 1):
        diff = top_diff
        for layer in reversed(self.layers):
            diff = layer.backward(diff)
        return diff
    
    # train using gradient descent
    def train_gd(self, X, Y, lr, iters=50000, debug=True):
        index = 0
        count = 0
        for _ in range(0, iters):
            input, expected = X[index], Y[index]
            self.layers[-1].target = expected

            loss = self.forward_pass(input)
            self.backward_pass()
            for layer in self.layers:
                if hasattr(layer, 'update_params'):
                    layer.update_params(lr)

            index = (index + 1) % len(X)
            if index == 0:
                if debug:
                    print("Epoch passed")

    def accuracy(self, X, Y):
        correct = 0
        for i in range(0, len(X)):
            input, expected = X[i], Y[i]
            output = self.forward_pass(input, self.layers[:-1])
            if output.argmax() == expected.argmax():
                correct += 1
        return float(correct) / len(X)
