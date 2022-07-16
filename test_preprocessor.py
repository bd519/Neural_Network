import numpy as np

from part1_nn_lib import Preprocessor


def test_preprocessor():
    """
    """

    data = np.loadtxt('iris.dat')
    np.random.shuffle(data)

    # print(np.shape(data))

    processor = Preprocessor(data)

    normalized_dataset = processor.apply(data)
    reverted_dataset = processor.revert(normalized_dataset)

    assert(np.allclose(reverted_dataset, data))
