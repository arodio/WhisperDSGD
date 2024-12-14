import random
import numpy as np
from igraph import Graph

import warnings


def get_weight_matrix(n_nodes, connectivity, seed, max_trials=100):
    """
    Generate a Metropolis-Hastings weight matrix for an Erdős–Rényi random graph.

    Parameters
    ----------
    n_nodes: int
    connectivity: float
    seed: int
    max_trials: int

    Returns
    -------
    weight_matrix: np.ndarray
    """
    random.seed(seed)
    min_connectivity = np.log(n_nodes) / n_nodes

    if connectivity < min_connectivity:
        msg = f"Connectivity {connectivity:.4f} is less than recommended {min_connectivity:.4f}"
        warnings.warn(msg, UserWarning)

    for _ in range(max_trials):
        graph = Graph.Erdos_Renyi(n=n_nodes, p=connectivity, directed=False)
        if graph.is_connected():
            adjacency_matrix = np.array(graph.get_adjacency().data, dtype=int)
            break
    else:
        raise ValueError(f"Failed to generate a connected graph after {max_trials} trials.")

    # Generate Metropolis-Hastings weights
    degrees = adjacency_matrix.sum(axis=1)
    weight_matrix = np.zeros_like(adjacency_matrix, dtype=float)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adjacency_matrix[i, j] == 1:
                weight_matrix[i, j] = 1 / (max(degrees[i], degrees[j]) + 1)
        weight_matrix[i, i] = max(0, 1 - weight_matrix[i].sum())

    return weight_matrix


def get_laplacian(weight_matrix):
    """
    Compute the graph Laplacian matrix from the weight matrix.
    Parameters
    ----------
    weight_matrix: np.ndarray

    Returns
    -------
    laplacian_matrix: np.ndarray
    """
    # Get the adjacency matrix
    adjacency_matrix = (weight_matrix > 0).astype(int)
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

    # Remove self-loops
    np.fill_diagonal(adjacency_matrix, 0)

    # Compute the degree matrix
    degrees = adjacency_matrix.sum(axis=1)
    degree_matrix = np.diag(degrees)

    # Compute the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    return laplacian_matrix
