import numpy as np
from scipy.spatial.distance import pdist, squareform


atomic_symbol_to_number_dict = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
}


def atomic_symbols_to_numbers(symbols):
    return np.array(
        [atomic_symbol_to_number_dict[symbol.upper()] for symbol in symbols]
    ).reshape(-1, 1)


def get_distance_matrix(xyz):
    """
    Get a distance matrix of distances between atoms

    Parameters:
    xyz (np.ndarray): array shape (n_atom, 3) of atomic coordinates

    Returns:
    distance matrix of type np.ndarray with shape (n_atoms, n_atoms)
    """
    return squareform(pdist(xyz))


def get_receivers_senders(xyz, cutoff, self_loops=False):
    """
    Get the two arrays receivers and senders
    Defines the graph connection by receiving
    and sending atom indices

    Parameters:
    xyz (np.ndarray): array shape (n_atom, 3) of atomic coordinates
    cutoff (int): No edge between atoms separated by dis. > cutoff

    Returns:
    receivers (np.ndarray): Shape (n_edges, 1), sorted
    senders (np.ndarray): Shape (n_edges, 1)
    """
    distance_matrix = get_distance_matrix(xyz)
    edges = np.argwhere(distance_matrix <= cutoff)

    if not self_loops:
        argwhere = np.argwhere(edges[:, 0] != edges[:, 1]).flatten()
        edges = edges[argwhere]

    return edges[:, 0].reshape(-1, 1), edges[:, 1].reshape(-1, 1)


def get_atom_distances(distance_matrix, receivers, senders):
    atom_distances = distance_matrix[receivers, senders].flatten()
    return atom_distances
