from __future__ import division
from __future__ import print_function
import os
import numpy as np

def read_glove_vectors(glove_vector_path):
    '''Method to read glove vectors and return an embedding dict.'''
    embeddings_index = {}
    with open(glove_vector_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs[:]
    return embeddings_index


def read_input_data(input_data_path):
    '''Method to read data from input_data_path'''
    #texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    texts = list(open(os.path.join(input_data_path, 'en_msd'), 'r').readlines())

    with open(os.path.join(input_data_path, 'en_msd_label'), 'r') as label_f:
        for line in label_f:
            label = int(line.strip())
            labels.append(label)

    return texts, labels

def train_data(data, labels):
    c1 = []
    c2 = []
    c3 = []
    l1 = []
    l2 = []
    l3 = []
    for i in range(data.shape[0]):
        if labels[i] == 0:
            c1.append(data[i])

        if labels[i] == 1:
            c2.append(data[i])

        else:
            c3.append(data[i])

    mono_num = len(c1) - 1
    swap_num = len(c2) - 1
    disc_num = len(c3) - 1
    mono_rand = []
    swap_rand = []
    disc_rand = []

    for i in range(450000):
        n = np.random.randint(0, mono_num)
        mono_rand.append(c1[n])
        l1.append(int(0))

    for i in range(450000):
        n = np.random.randint(0, swap_num)
        swap_rand.append(c2[n])
        l2.append(int(1))

    for i in range(450000):
        n = np.random.randint(0, disc_num)
        disc_rand.append(c3[n])
        l3.append(int(2))

    x_train = mono_rand + swap_rand + disc_rand
    y_train = l1 + l2 + l3

    return x_train, y_train

