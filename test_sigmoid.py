import numpy as np

from part1_nn_lib import LinearLayer
from part1_nn_lib import SigmoidLayer
from part1_nn_lib import MSELossLayer

from load_test_batches import load_test_batches


def test_linear():

    batches = load_test_batches()

    # Create a linear layer, activation layer, and a loss layer
    layer = LinearLayer(n_in=4, n_out=3)
    activation = SigmoidLayer()
    loss_layer = MSELossLayer()

    # For each batch, run the forward, backward, and parameter update functions
    for (input_batch, output_batch) in batches:

        # Run the forward propagaion
        intermediate = layer.forward(input_batch)
        outputs = activation.forward(intermediate)

        # Compute loss gradient
        loss_layer.forward(outputs, output_batch)
        current_loss_grad = loss_layer.backward()

        # Perform backwards pass to compute gradients
        intermediate_grad = activation.backward(current_loss_grad)
        layer.backward(intermediate_grad)

        # Update parameters for gradient descent
        layer.update_params(0.01)
        activation.update_params(0.01)
