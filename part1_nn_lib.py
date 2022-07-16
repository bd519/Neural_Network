import numpy as np
import pickle

import math


def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """

    print('xavier_init')
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)
    print('xavier_init done')


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Sigmoid layer.
        """
        print('sigmoid constructor')
        self._cache_current = None
        print('sigmoid constructor done')

    @staticmethod
    def _sigmoid(x):
        """ Sigmoid function

        Computes the element-wise sigmoid of the input array.

        Aruguments:
            x {numpy.ndarray} -- Input array

        Returns:
            {numpy.ndarray} -- Output array
        """

        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        print('sigmoid forward')
        # Cache the input in... the cache
        self._cache_current = x
        # Evaluate and return the sigmoid'd input array
        sigmoid_x = 1 / (1 + np.exp(-x))
        print('sigmoid forward done')
        return sigmoid_x


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        print('sigmoid backward')
        # Perform a vectorized element-wise operation on the cached inputs to
        # evaluate the activation derivative
        activated_array = 1 / (1 + np.exp(-(self._cache_current)))
        activation_derivative = np.multiply(activated_array, 1 - activated_array)

        # Multiply the derivative and the gradient element-wise, returning the
        # result
        result = np.multiply(grad_z, activation_derivative)
        print('sigmoid backward done')
        return result


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        print('relu constructor')
        self._cache_current = None
        print('relu constructor done')

    def forward(self, x):
        """
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        print('relu forward')
        self._cache_current = x

        result = x * (x > 0)
        print('relu forward done')
        return result

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        print('relu backward')
        # Perform a vectorized element-wise operation on the cached inputs to
        # evaluate the activation derivative
        operation = lambda i: 1 if i > 0 else 0
        activation_derivative = np.vectorize(operation)(self._cache_current)

        # Multiply the derivative and the gradient element-wise, returning the
        # result
        result = np.multiply(grad_z, activation_derivative)
        print('relu backward done')
        return result

class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """

        print('linear constructor')
        self.n_in = n_in
        self.n_out = n_out

        weight_values = xavier_init(n_in * n_out)
        self._W = np.array_split(weight_values, n_in)
        self._b = np.zeros(n_out)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None
        print('linear constructor done')

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        print('linear forward')
        # Cache inputs for backpropagation
        self._cache_current = x

        # Create an expanded biases matrix
        batch_size, _ = np.shape(x)
        reshaped_biases = np.tile(self._b, (batch_size, 1))

        # Calculate forward propagation result
        weighted_nodes = np.matmul(x, self._W)
        result = np.add(weighted_nodes, reshaped_biases)
        print('linear forward done')

        return result

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        print('linear backward')
        # Compute the gradient of the weights
        x_transpose = np.transpose(self._cache_current)
        self._grad_W_current = np.matmul(x_transpose, grad_z)

        # Compute the gradient of the biases
        batch_size, _ = np.shape(grad_z)
        ones_vector = np.ones((1, batch_size))
        self._grad_b_current = np.matmul(ones_vector, grad_z)

        # Calculate layer input gradients, and return result
        W_transpose = np.transpose(self._W)
        result = np.matmul(grad_z, W_transpose)
        print('linear backward done')
        return result

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        print('linear update_params')
        # Update both the weights and biases parameters according to the
        # previously computed gradients
        self._W = self._W - np.multiply(learning_rate, self._grad_W_current)
        self._b = self._b - np.multiply(learning_rate, self._grad_b_current)
        print('linear update_params done')


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers.
            - activations {list} -- List of the activation functions to apply
                to the output of each linear layer.
        """

        print('multilayer constructor')
        print(f'  creating multi-layer network')
        print(f'    inputs, neurons, activations: {input_dim}, {neurons}, {activations}')
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        self._layers = []

        # Construct an instance of Linear layer for each entry in the neurons
        # list, optionally attaching an activation function layer on top
        n_out = input_dim
        for index in range(len(neurons)):

            # Update the layer dimensions
            n_in = n_out
            n_out = neurons[index]

            # Instantiate and append the layer
            self._layers.append(LinearLayer(n_in, n_out))

            # >>>>

            # Continue if no activation function has been specified
            # activation = activations[index]
            # if not activation or activation == 'identity':
            #     continue

            # Add the activation function layer, if it's been specified
            # activation_classes = {
            #     'relu': ReluLayer,
            #     'sigmoid': SigmoidLayer
            # }

            # Ignore invalid activation class names, and if valid, add the
            # relevant class to the network
            # if activation not in activation_classes:
            #     continue
            # self._layers.append(activation_classes[activation]())

            # >>>>

            if activations[index] == 'relu':
                self._layers.append(ReluLayer())
            elif activations[index] == 'sigmoid':
                self._layers.append(SigmoidLayer())

            # <<<<
        print('multilayer constructor done')

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """

        print('multilayer forward')
        for layer in self._layers:
            x = layer.forward(x)
        print('multilayer forward done')
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """

        print('multilayer backward')
        for layer in reversed(self._layers):
            grad_z = layer.backward(grad_z)
        print('multilayer backward done')
        return grad_z

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        print('multilayer update_params')
        for layer in self._layers:
            layer.update_params(learning_rate)
        print('multilayer update_params done')


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag):

        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        print(f'network details: {network.input_dim}, {network.neurons}, {network.activations}')

        print('creating instance of trainer class')
        print(f'  shuffle, batch size, epochs, loss function: {shuffle_flag}, {batch_size}, {nb_epoch}, {loss_fun}')

        # >>>>

        # loss_classes = {
        #     'mse': MSELossLayer,
        #     'bce': CrossEntropyLossLayer,
        #     'cross_entropy': CrossEntropyLossLayer,
        # }
        # self._loss_layer = loss_classes[loss_fun]()

        # >>>>

        if loss_fun == 'mse':
            self._loss_layer = MSELossLayer()
        else:
            self._loss_layer = CrossEntropyLossLayer()

        # <<<<

        print('trainer constructor finished')

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns:
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """

        print('trainer shuffle')

        # Record the number of features, for splitting the shuffled dataset
        _, n_features = np.shape(input_dataset)

        # Concatenate the inputs, targets -- and shuffle the union
        union = np.concatenate((input_dataset, target_dataset), axis=1)
        np.random.shuffle(union)

        # Split the shuffled union back into its components
        shuffled_inputs = union[:, :n_features]
        shuffled_outputs = union[:, n_features:]

        print('trainer shuffle done')
        # Return the shuffled subarrays
        return (shuffled_inputs, shuffled_outputs)

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """

        print(f'training: {np.shape(input_dataset)}, {np.shape(target_dataset)}')

        # Loop for the specified nb_epoch times
        for index in range(self.nb_epoch):

            # Shuffle, if necessary
            if self.shuffle_flag:
                input_dataset, target_dataset = \
                        Trainer.shuffle(input_dataset, target_dataset)

            # Isolate the input columns, and memoize their size
            data_point_count, _ = np.shape(input_dataset)

            # Split the dataset into batches of `batch_size` elements
            batches = []
            batch_count = int(math.ceil(data_point_count / self.batch_size))
            for batch_index in range(batch_count):
                start = batch_index * self.batch_size
                end = start + self.batch_size
                if end > data_point_count:
                    break

                input_batch = input_dataset[start:end]
                target_batch = target_dataset[start:end]
                batches.append((input_batch, target_batch))

            # For each batch: perform forward pass, backward pass, and gradient
            # descent
            for (current_input_batch, current_target_batch) in batches:

                # Perform forward pass
                current_outputs = self.network.forward(current_input_batch)

                # Compute loss gradient
                current_loss = self._loss_layer.forward(current_outputs, current_target_batch)
                current_loss_grad = self._loss_layer.backward()

                # Perform backwards pass to compute gradients
                self.network.backward(current_loss_grad)

                # Update parameters for gradient descent
                self.network.update_params(self.learning_rate)
            
            if (index % 10) == 0:
                print(f"@ epoch {index} the loss is {current_loss}")

        print(' -> Training complete')

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        """

        print('trainer eval_loss')

        # Do a forward pass of inputs through the network
        outputs = self.network.forward(input_dataset)

        # Compute loss function of results compared to target outputs
        loss = self._loss_layer.forward(outputs, target_dataset)

        print(f' -> eval_loss computed: {loss}')

        return loss


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """

        print('preprocessor constructor')

        self.a = 0
        self.b = 1

        self.column_maxima = np.max(data, axis=0)
        self.column_minima = np.min(data, axis=0)

        print('preprocessor construtor done')

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """

        print('preprocessor apply')

        result = data.copy()

        # For each column, substract the minimum of that column to each elements
        result = result - self.column_minima

        # Multiply the matrix by the scalar b-a
        result = np.multiply(result, (self.b - self.a))

        # Divide each element by the Maximum-Minimum of their column
        result = result / (self.column_maxima - self.column_minima)

        # Add a to the whole matrix
        result = np.add(self.a, result)

        print('preprocessor apply done')

        return result

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """

        print('preprocessor revert')

        result = data.copy()

        result = np.subtract(result, self.a)

        result = result * (self.column_maxima - self.column_minima)

        result = np.divide(result, (self.b - self.a))

        result = result + self.column_minima

        print('preprocessor revert done')

        return result


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
