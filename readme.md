# Artificial Neural Networks

#### Introduction to Machine Learning Coursework, Autumn 2021

## Part 1: Neural Network Mini-Library

### Overview

`par1_nn_lib.py` is a self-contained neural network library, supporting several combinations of [activation](https://en.wikipedia.org/wiki/Activation_function) and [loss](https://en.wikipedia.org/wiki/Loss_function) functions, as well as a built-in data pre-processor and training class.

#### Activation functions

Layer activation function | Description
--------------------------|---
`identity`                | output remains the same as the input
`relu`                    | [rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
`sigmoid`                 | [logistic function](https://en.wikipedia.org/wiki/Logistic_function)

#### Loss functions

Network loss function    | Description
-------------------------|---
`mse`                    | [mean-square error](https://en.wikipedia.org/wiki/Mean_squared_error) loss
`bce` or `cross_entropy` | computes [softmax](https://en.wikipedia.org/wiki/Softmax_function) and negative [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) loss

### Usage

#### Creating a Single Layer

``` python
from part1_nn_lib import LinearLayer, MSELossLayer


# Define layer input/output dimensions, and the learning rate
inputs, outputs = 4, 3
learning_rate = 0.01

# Create a linear layer, and a mean-square error loss layer
linear_layer = LinearLayer(n_in=inputs, n_out=outputs)
loss_layer = MSELossLayer()

# TODO: Load your dataset here into the arrays input_dataset and target_dataset

# Propagate forwards through the layer
forward_output = linear_layer.forward(input_dataset)

# Calculate the loss gradient
loss_layer.forward(forward_output, target_dataset)
loss_gradient = loss_layer.backward()

# Propagate backwards through the layer, and update parameters
linear_layer.backward(loss_gradient)
linear_layer.update_params(learning_rate)
```

The snippet above allows you to create a single layer, and perform a single epoch of training. Here the layer is linear, but you could also add and propagate through a `RELULayer` or `SigmoidLayer` for a different activation function.

Note that you could also use an instance of `CrossEntropyLossLayer` rather than `MSELossLayer`.

#### Creating a Multi-Layer Network

``` python
from part1_nn_lib import MultiLayerNetwork, MSELossLayer

# Define the number of network inputs/outputs, the size and number of hidden
# layers, and the activation functions of those layers
# NOTE: here you can add as many layers of whatever size and activation
# function you want
inputs = 4
layer_neurons = [16, 3]
layer_activations = ['relu', 'identity']

# Create the network, and a mean-square error loss layer
network = MultiLayerNetwork(inputs, layer_neurons, layer_activations)
loss_layer = MSELossLayer()

# TODO: Load your dataset here into the arrays input_dataset and target_dataset

# Propagate forwards through the network
forward_output = network.forward(input_dataset)

# Calculate the loss gradient
loss_layer.forward(forward_output, target_dataset)
loss_gradient = loss_layer.backward()

# Propagate backwards through the network, and update parameters
network.backward(loss_gradient)
network.update_params(learning_rate)
```

Here we're using a `MultiLayerNetwork` to abstract the creation of individual layers, but the training process remains much the same.

To change the number of hidden layers, or the number of nodes within those layers, `layer_neurons` can be updated. Similarly, to change the corresponding activation functions, alter `layer_activations`. Note that the final value in `layer_neurons` (in this case, 3) should be the number of elements in your target dataset.

#### Propagating Using the Trainer Class

``` python
from part1_nn_lib import MultiLayerNetwork, Trainer

# Define the number of network inputs/outputs, the size and number of hidden
# layers, and the activation functions of those layers
inputs = 4
layer_neurons = [16, 3]
layer_activations = ['relu', 'identity']

# Create a network
network = MultiLayerNetwork(inputs, layer_neurons, layer_activations)

# Create an instance of the trainer class, where:
#   - Each batch propagated in the network has (8) elements
#   - The trainer runs for (100) propagation epochs
#   - Each epoch updates parameters with a learning rate of (0.01)
#   - The dataset (is) shuffled before each epoch
#   - The loss function used is (mse)
trainer = Trainer(network=network,
    batch_size=8,
    nb_epoch=100,
    learning_rate=0.01,
    shuffle_flag=True,
    loss_fun='mse')

# TODO: Load your dataset here into the arrays input_dataset and target_dataset

trainer.train(input_dataset, output_dataset)
```

Here we're using an instance of the `Trainer` class to run many epochs of training on a network. You can see how, in the constructor, we're setting the parameters for this training. You may also wish to normalize your data before using it for training, which can be done using the `Preprocessor` class:

```python

# TODO: Load your dataset into the array `dataset`

processor = Preprocessor(dataset)

normalized_dataset = processor.apply(data)
```

## Part 2: Neural Network

### Installation guide

#### Compatibility

This code was developed and tested on Ubuntu 20.04 using Python 3.8.10.

It relies on the following dependencies:
 - NumPy
 - Pandas Dataframe
 - PyTorch
 - Sklearn

### Installation process

#### Getting the codebase

In order to use the codebase, it first needs to be cloned to your local machine using `$ git clone https://gitlab.doc.ic.ac.uk/lab2122_autumn/Neural_Networks_38.git` in the parent folder you want the codebase to be in.

You can then enter the project folder using `$ cd Neural_Networks_38`.

<br />

# Part 2: Create and train a neural network for regression

<br />

## Code implementation and optimisation

### Running the code and applying it to a test set

The Neural Network Model is based around a Regressor Class which inherits its behaviour from the pytorch neural network library.

```python
class Regressor(nn.Module):
```

To use a naive Neural Network model simply create an instance of the neural network. The default configuration for any neural network is not be optimised for any particular dataset and on instantiation, the model is not trained. The default structure is three layers consisting of 60, 40 and 50 neurons. The epoch limit is 1000 with a batch_size of 128 and learning rate of 0.01.

```python
def __init__(self, x=None, nb_epoch = 1000, batch_size = 128, learning_rate = 0.01, nb_overfitting_epoch_threshold =30, layer_sizes = [60,40,50]):
```
### Training a network

The neural network can be trained for a particular dataset using the fit member function with the desired input and output. An example is shown below:

```python
output_label = "median_house_value"

# Use pandas to read CSV data as it contains various object types
# Feel free to use another CSV reader tool
# But remember that LabTS tests take Pandas Dataframe as inputs
data = pd.read_csv("housing.csv")
train, test = sklearn.model_selection.train_test_split(data, test_size=0.2)
# Spliting input and output
x_train = train.loc[:, train.columns != output_label]
y_train = train.loc[:, [output_label]]
r = Regressor(x=x_train, batch_size=10, learning_rate=0.075, layer_sizes=[10,20])
r.fit(x_train,y_train)
```
### Predicting and Scoring with the Network

The member functions score and predict can be used for scoring a neural network and predicting outcomes. Score takes as arguments the input and expected output as pandas dataframes. Score then returns a float representing MSE loss. Predict also requires the input  data as a pandas dataframe and returns a numpy array of non-normalised predicted values.

### Tuning the Hyper-Parameters

The fit function does not modify any hyper-parameters. To fit and tune hyper-parameters, call the RegressorHyperParameterSearch function with the file location pointing towards the input csv dataset. The function will return an optimised model.

```python
def RegressorHyperParameterSearch(file_path)
```

### Loading and Saving a Model

To save/load a model simply call the function save_regressor(regressor_object)/ load_regressor()

The regressor is saved in multiple parts in the following files:

- scaler_x.save
- scaler_y.save
- reg_struct_part_2_model.json
- part2_model.torch

Shown below is an example of loading/saving a model:

```python
result = RegressorHyperParameterSearch()
save_regressor(result)

solution = load_regressor()
data = pd.read_csv("housing.csv")
train, test = sklearn.model_selection.train_test_split(data, test_size=0.2)
```
