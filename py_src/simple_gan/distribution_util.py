import numpy as np


def build_distribution(dimensions=[1], apply_transformation=lambda s: s):

    def get_samples(size):
        random_distribution = np.random.rand(size, *dimensions)
        for idx in range(size):
            random_distribution[idx] = apply_transformation(random_distribution[idx])

        return random_distribution

    return get_samples
