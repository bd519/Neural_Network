from unittest import TestCase
import numpy as np

from part1_nn_lib import LinearLayer
from part1_nn_lib import MSELossLayer

from load_test_batches import load_test_batches


def test_linear():
    """
    """

    batches = load_test_batches()

    n_in = 4
    n_out = 3

    # Create a linear layer, and a loss layer
    layer = LinearLayer(n_in=n_in, n_out=n_out)
    loss_layer = MSELossLayer()

    # For each batch, run the forward, backward, and parameter update functions
    for (input_batch, output_batch) in batches:
        batch_size = len(input_batch)

        # Run the forward propagaion
        outputs = layer.forward(input_batch)
        forward_shape = np.shape(outputs)
        assert forward_shape == (batch_size, n_out)

        # Compute loss gradient
        loss_layer.forward(outputs, output_batch)

        current_loss_grad = loss_layer.backward()

        # Perform backwards pass to compute gradients
        backward_output = layer.backward(current_loss_grad)
        backward_shape = np.shape(backward_output)
        assert backward_shape == (batch_size, n_in)

        # Update parameters for gradient descent
        layer.update_params(0.01)
