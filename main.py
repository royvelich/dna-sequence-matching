import os
import numpy
from matplotlib import pyplot
from numpy.random import default_rng
from scipy.stats import wasserstein_distance
import matplotlib
import matplotlib.pyplot as plt
import Bio
import Bio.SeqRecord
import Bio.SeqIO
import pygtrie
import re


def get_subfragment_indices_by_length(fragment, subfragment_min_length):
    while True:
        fragment_indices_count = fragment.shape[0]
        start_index = numpy.random.randint(fragment_indices_count)
        for end_index in range(start_index, fragment_indices_count):
            if fragment[end_index] - fragment[start_index] > subfragment_min_length:
                indices = numpy.array(list(range(start_index, end_index+1)))
                return indices


def get_subfragment_indices_by_count(fragment, subfragment_indices_count):
    while True:
        fragment_indices_count = fragment.shape[0]
        start_index = numpy.random.randint(fragment_indices_count)
        end_index = start_index + subfragment_indices_count + 1
        if end_index < fragment_indices_count:
            indices = numpy.array(list(range(start_index, end_index + 1)))
            return indices


def match_fragments(full_fragment, partial_fragment):
    full_fragment_length = numpy.max(full_fragment)
    partial_fragment_length = numpy.max(partial_fragment)
    start_pos = 0
    end_pos = partial_fragment_length
    delta = 0.00001
    max_dist = numpy.inf
    matched_indices = None
    x = []
    y = []
    while end_pos < 1:
        # indices = numpy.where(numpy.any((full_fragment > start_pos) and (full_fragment < end_pos)))
        # indices = numpy.where(numpy.any(full_fragment > start_pos))
        index_predicates = (full_fragment > start_pos) & (full_fragment < end_pos)
        indices = numpy.where(index_predicates)[0]
        indices.sort()

        current_fragment = full_fragment[indices]
        current_fragment = current_fragment - current_fragment[0]

        dist = wasserstein_distance(current_fragment, partial_fragment)
        x.append(start_pos)
        y.append(dist)
        if dist < max_dist:
            max_dist = dist
            matched_indices = indices

        end_pos = end_pos + delta
        start_pos = start_pos + delta

    return matched_indices, numpy.array(x), numpy.array(y)


def dye_chromosome(chromosome: Bio.SeqRecord.SeqRecord, fluorochrome: str):
    chromosome = chromosome.lower()
    fluorochrome = fluorochrome.lower()
    chromosome_str = str(chromosome.seq)
    indices = [m.start() for m in re.finditer(fluorochrome, chromosome_str)]
    fragment = numpy.array(indices).astype(float) / float(len(chromosome_str))
    return indices, fragment


def plot_fragment(fragment, color, markersize=2):
    y = numpy.zeros_like(fragment)
    x = fragment
    plt.plot(x, y, 'o', markersize=markersize, markerfacecolor=color, markeredgecolor=color)


def plot_fragments(full_fragment, partial_fragment, sampled_partial_fragment, transformed_sampled_partial_fragment, transformed_noised_sampled_partial_fragment, matched_fragment, x_dist, y_dist):
    plt.figure(figsize=(40, 2))
    plot_fragment(fragment=full_fragment, color='blue')
    plot_fragment(fragment=partial_fragment, color='red')
    plt.xlim(0, 1)
    plt.show()

    # y3 = numpy.zeros_like(sampled_partial_fragment)
    # x3 = sampled_partial_fragment
    # plt.plot(x3, y3, 'o', markersize=2, markerfacecolor='green', markeredgecolor='green')


    plt.figure(figsize=(40, 2))
    plot_fragment(fragment=sampled_partial_fragment, color='green')
    plt.xlim(0, 1)
    plt.show()

    plt.figure(figsize=(40, 2))
    plot_fragment(fragment=transformed_sampled_partial_fragment, color='magenta')
    plt.xlim(0, 1)
    plt.show()

    plt.figure(figsize=(40, 2))
    plot_fragment(fragment=transformed_noised_sampled_partial_fragment, color='orange')
    plt.xlim(0, 1)
    plt.show()

    plt.figure(figsize=(40, 2))
    plot_fragment(fragment=matched_fragment, color='purple')
    plt.xlim(0, 1)
    plt.show()

    plt.figure(figsize=(40, 10))
    plt.plot(x_dist, y_dist, '-', markersize=2)
    plt.xlim(0, 1)


if __name__ == '__main__':
    filename = os.path.normpath("C:/genome/GCA_000001405.29_GRCh38.p14_genomic.fna")
    seq_dict = {rec.id: rec for rec in Bio.SeqIO.parse(filename, "fasta")}
    chromosome4 = seq_dict['CM000666.2']
    chromosome4_segment = chromosome4[88227200:88704008]
    chromosome4_segment_len = len(chromosome4_segment.seq)
    print(f'chromosome4_segment length: {chromosome4_segment_len}')
    full_fragment_indices, full_fragment = dye_chromosome(chromosome=chromosome4_segment, fluorochrome='CTTAAG')

    # size = 100
    # rng = default_rng(seed=0)
    # full_fragment = rng.uniform(size=size)
    # full_fragment.sort()

    # partial_fragment_indices = get_subfragment_indices_by_length(fragment=full_fragment, subfragment_min_length=0.1)

    experiments_count = 10
    for i in range(experiments_count):
        print(f'========= Experiment {i} =========')

        partial_fragment_indices = get_subfragment_indices_by_count(fragment=full_fragment, subfragment_indices_count=16)
        partial_fragment = full_fragment[partial_fragment_indices]

        sampling_factor = 0.9
        sampled_partial_fragment_size = int(partial_fragment_indices.shape[0] * sampling_factor)
        sampled_partial_fragment_meta_indices = numpy.random.choice(partial_fragment_indices.shape[0], sampled_partial_fragment_size, replace=False)
        sampled_partial_fragment_meta_indices.sort()

        sampled_partial_fragment_indices = partial_fragment_indices[sampled_partial_fragment_meta_indices]
        sampled_partial_fragment = full_fragment[sampled_partial_fragment_indices]
        sampled_partial_fragment.sort()

        min_scale = 0.95
        max_scale = 1.05
        scale = (max_scale - min_scale) * numpy.random.random(1) + min_scale

        mu, sigma = 0, 0.001
        transformed_sampled_partial_fragment = scale * (sampled_partial_fragment - sampled_partial_fragment[0])
        noise = numpy.random.normal(mu, sigma, transformed_sampled_partial_fragment.shape[0])
        print(f'noise: {noise}')
        print(f'noise (bases): {noise * chromosome4_segment_len}')

        transformed_noised_sampled_partial_fragment = transformed_sampled_partial_fragment + noise

        print(f'full fragment length: {full_fragment.shape[0]}')
        print(f'partial fragment length: {transformed_sampled_partial_fragment.shape[0]}')

        matched_indices, x_dist, y_dist = match_fragments(full_fragment=full_fragment, partial_fragment=transformed_noised_sampled_partial_fragment)
        matched_fragment = full_fragment[matched_indices]

        print(f'matched indices: {matched_indices}')
        print(f'partial fragment indices: {partial_fragment_indices}')

        indices_diff = len(list(set(matched_indices) - set(partial_fragment_indices)))
        indices_count = len(list(partial_fragment_indices))
        if indices_diff > indices_count:
            print(f'matching_ratio: FAILED')
        else:
            matching_ratio = 1 - (indices_diff / indices_count)
            print(f'matching_ratio: {matching_ratio}')

        plot_fragments(
            full_fragment=full_fragment,
            partial_fragment=partial_fragment,
            sampled_partial_fragment=sampled_partial_fragment,
            transformed_sampled_partial_fragment=transformed_sampled_partial_fragment,
            transformed_noised_sampled_partial_fragment=transformed_noised_sampled_partial_fragment,
            matched_fragment=matched_fragment,
            x_dist=x_dist,
            y_dist=y_dist)

