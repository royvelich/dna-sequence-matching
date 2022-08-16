import numpy
from matplotlib import pyplot
from numpy.random import default_rng
from scipy.stats import wasserstein_distance


def get_fragment_indices(fragment, min_length):
    while True:
        indices_count = fragment.shape[0]
        start_index = numpy.random.randint(indices_count)
        for end_index in range(start_index, indices_count):
            if fragment[end_index] - fragment[start_index] > min_length:
                indices = numpy.array(list(range(start_index, end_index+1)))
                return indices


def match_fragments(full_fragment, partial_fragment):
    full_fragment_length = numpy.max(full_fragment)
    partial_fragment_length = numpy.max(partial_fragment)
    start_pos = 0
    end_pos = partial_fragment_length
    delta = 0.00001
    max_dist = numpy.inf
    matched_indices = None
    while end_pos < 1:
        # indices = numpy.where(numpy.any((full_fragment > start_pos) and (full_fragment < end_pos)))
        # indices = numpy.where(numpy.any(full_fragment > start_pos))
        index_predicates = (full_fragment > start_pos) & (full_fragment < end_pos)
        indices = numpy.where(index_predicates)[0]
        indices.sort()

        current_fragment = full_fragment[indices]
        current_fragment = current_fragment - current_fragment[0]

        dist = wasserstein_distance(current_fragment, partial_fragment)
        if dist < max_dist:
            max_dist = dist
            matched_indices = indices

        end_pos = end_pos + delta
        start_pos = start_pos + delta

    return matched_indices


if __name__ == '__main__':
    size = 1000
    rng = default_rng(seed=0)
    full_fragment = rng.uniform(size=size)
    full_fragment.sort()

    partial_fragment_indices = get_fragment_indices(fragment=full_fragment, min_length=0.1)
    sampling_factor = 0.7
    sampled_partial_fragment_size = int(partial_fragment_indices.shape[0] * sampling_factor)
    sampled_partial_fragment_meta_indices = numpy.random.choice(partial_fragment_indices.shape[0], sampled_partial_fragment_size, replace=False)
    sampled_partial_fragment_meta_indices.sort()

    sampled_partial_fragment_indices = partial_fragment_indices[sampled_partial_fragment_meta_indices]
    sampled_partial_fragment = full_fragment[sampled_partial_fragment_indices]
    sampled_partial_fragment.sort()

    min_scale = 0.95
    max_scale = 1.05
    scale = (max_scale - min_scale) * numpy.random.random(1) + min_scale
    # scale = 1

    transformed_sampled_partial_fragment = scale * (sampled_partial_fragment - sampled_partial_fragment[0])

    matched_indices = match_fragments(full_fragment=full_fragment, partial_fragment=transformed_sampled_partial_fragment)

    h = 5

    # y = x[:50:2]
    # z = 1.05 * y - y[0]
    # # y = y + .06
    # z = z + rng.normal(size=len(z)) * 1e-4
    # # y = rng.uniform(size=50)
    # z.sort()
    #
    # X, Z = numpy.meshgrid(x, z, indexing='ij')
    # pyplot.figure(0, clear=True)
    # pyplot.plot(X.ravel(), Z.ravel(), '.')
    #
    # pyplot.figure(1, clear=True)
    # pyplot.plot((Z - X).ravel(), Z.ravel(), '.')
    # pyplot.xlim(-.1, .1)

    d1 = wasserstein_distance([0, 1, 3], [5, 6, 8])
    d2 = wasserstein_distance([0, 1, 3], [5, 6, 8, 0])
    j = 6
