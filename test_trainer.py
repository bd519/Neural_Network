import numpy as np

from part1_nn_lib import Trainer
from part1_nn_lib import MultiLayerNetwork

def test_trainer():
    """
    """

    network = MultiLayerNetwork(4, [16, 3], ['relu', 'sigmoid'])

        # Load, shuffle the dataset
    data = np.loadtxt('iris.dat')
    np.random.shuffle(data)

    # Split into an input and output batch
    inputs = data[:, :4]
    outputs = data[:, 4:]

    trainer = Trainer(network=network,
        batch_size=8,
        nb_epoch=100,
        learning_rate=0.01,
        shuffle_flag=True,
        loss_fun='mse')

    trainer.train(inputs, outputs)
