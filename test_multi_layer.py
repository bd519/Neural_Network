import numpy as np

from part1_nn_lib import MSELossLayer
from part1_nn_lib import MultiLayerNetwork

from load_test_batches import load_test_batches


def test_multi_layer():
    """
    """

    batches = load_test_batches()

    loss_layer = MSELossLayer()
    network = MultiLayerNetwork(4, [16, 3], ['relu', 'identity'])

    for (current_input_batch, current_target_batch) in batches:

        # Perform forward pass
        current_outputs = network.forward(current_input_batch)

        # Compute loss gradient
        loss_layer.forward(current_outputs, current_target_batch)
        current_loss_grad = loss_layer.backward()

        # Perform backwards pass to compute gradients
        network.backward(current_loss_grad)

        # Update parameters for gradient descent
        network.update_params(0.01)
