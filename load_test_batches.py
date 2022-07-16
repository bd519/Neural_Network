import math

import numpy as np


def load_test_batches(batch_size: int = 8):

    # Load, shuffle the dataset
    data = np.loadtxt('iris.dat')
    np.random.shuffle(data)

    # Split into an input and output batch
    inputs = data[:, :4]
    outputs = data[:, 4:]

    # Calculate some batch-size related stuff
    batch_size = 8
    data_point_count, _ = np.shape(inputs)
    batch_count = int(math.ceil(data_point_count / batch_size))

    # Split each batch into input and output
    batches = []
    for index in range(batch_count):
        start = index * batch_size
        end = max((start + batch_size), data_point_count)

        input_batch = inputs[start:end]
        output_batch = outputs[start:end]
        batches.append((input_batch, output_batch))

    return batches
