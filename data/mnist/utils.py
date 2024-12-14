import random
import time

import numpy as np

import os
import torch

from torch.utils.data import ConcatDataset, Subset, DataLoader

def split_list_by_proportions(n, proportions):
    """

    Parameters
    ----------
    n : int
    proportions : list
    """
    assert np.isclose(sum(proportions), 1), "The sum of the proportions must be 1"

    elements = np.arange(n)
    indices = np.cumsum(np.array(proportions) * n).astype(int)
    sublists = np.split(elements, indices[:-1])

    return [list(sublist) for sublist in sublists]

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def iid_split(dataset, n_clients, frac, seed=1234):
    """
    split classification dataset among `n_clients` in an IID fashion. The dataset is split as follow:

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_clients: number of clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)
    rng.shuffle(selected_indices)

    return iid_divide(selected_indices, n_clients)


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_components, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follows:
        1) classes are grouped into `n_components`
        2) for each component `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_components: number of components to consider; if it is `-1`, then `n_components = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_components == -1:
        n_components = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    components_labels = iid_divide(all_labels, n_components) # TODO : check with dirichlet

    label2component = dict()  # maps label to its component
    for group_idx, labels in enumerate(components_labels):
        for label in labels:
            label2component[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    components_sizes = np.zeros(n_components, dtype=int)
    components = {k: [] for k in range(n_components)}
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2component[label]
        components_sizes[group_id] += 1
        components[group_id].append(idx)

    for _, component in components.items():
        rng.shuffle(component)

    clients_counts = np.zeros((n_components, n_clients), dtype=np.int64)  # number of samples by client from each component

    for component_id in range(n_components):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[component_id] = np.random.multinomial(components_sizes[component_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for component_id in range(n_components):
        component_split = split_list_by_indices(components[component_id], clients_counts[component_id])

        for client_id, indices in enumerate(component_split):
            clients_indices[client_id] += indices

    return clients_indices


def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follows:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    # TODO: use numpy generator only
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices


def by_labels_non_iid_split_with_proportions(dataset, n_classes, n_clients, n_components, alpha, frac, proportions, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follows:
        1) classes are grouped into `n_components`
        2) for each component `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_components: number of components to consider; if it is `-1`, then `n_components = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param proportions: list of proportions for each client
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_components == -1:
        n_components = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    components_labels = iid_divide(all_labels, n_components)

    label2component = dict()  # maps label to its component
    for group_idx, labels in enumerate(components_labels):
        for label in labels:
            label2component[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    components_sizes = np.zeros(n_components, dtype=int)
    components = {k: [] for k in range(n_components)}
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2component[label]
        components_sizes[group_id] += 1
        components[group_id].append(idx)

    for _, component in components.items():
        rng.shuffle(component)

    clients_counts = np.zeros((n_components, n_clients), dtype=np.int64)  # number of samples by client from each component

    for component_id in range(n_components):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        weights_with_proportions = weights * proportions
        weights_with_proportions = weights_with_proportions / weights_with_proportions.sum()
        clients_counts[component_id] = np.random.multinomial(components_sizes[component_id], weights_with_proportions)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for component_id in range(n_components):
        component_split = split_list_by_indices(components[component_id], clients_counts[component_id])

        for client_id, indices in enumerate(component_split):
            clients_indices[client_id] += indices

    return clients_indices


def save_clusters_data(dataset, clusters_indices, base_path):
    for i, indices in enumerate(clusters_indices):
        cluster_path = os.path.join(base_path, f'cluster_{i}')
        os.makedirs(cluster_path, exist_ok=True)

        # Create a subset of the dataset for the current cluster
        cluster_subset = Subset(dataset, indices)

        # Save the subset to disk
        torch.save(cluster_subset, os.path.join(cluster_path, 'dataset.pt'))

def load_cluster_data(base_path, cluster_id):
    cluster_path = os.path.join(base_path, f'cluster_{cluster_id}', 'dataset.pt')
    return torch.load(cluster_path)
