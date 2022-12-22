"""Class definition of data objects and datasets objects

class MolDict - defines the deta objects representing
individual and batched molecules. Keys that must be defined
are specified in "Data fields". (It is just a dictionary)

class MolDictList - defines the dataset object. (It is just
a list)

function batch_mol_dicts - takes a list of MolDict objects
as forms a bacth combining the MolDicts into one batched
MolDict

function save_dataset - saves the MolDictList in binary format

function load_dataset - loads the MolDictList from binary format
"""

import numpy as np
import pickle

# Data fields
# The following data fields are used to specify the graph molecular input
# All valuas are numpy arrays
ATOMS = "atoms"  # Int, shape=(n_atoms,1)
COORDINATES = "coordinates"  # Float,  shape=(n_atoms, 3)
ENERGY = "energy"  # Float, shape=(n_mols,1)

RECEIVERS = "receivers"  # Int, shape=(n_edges, 1) #Sorted low to high
SENDERS = "senders"  # Int, shape=(n_edges, 1)

N_ATOMS = "n_atoms"  # Int, shape=(n_mols,1)
N_EDGES = "n_edges"  # Int, shape=(n_mols,1)
BATCH_INDEX = "batch_index"  # Int, shape=(n_atoms,1)
INDEX = "index"  # Int, shape=(n_mols,1) #Bookkeeping only
MOLECULE_INDEX = "molecule_index"  # Int, shape=(n_mols,1) #Bookkeeping only

# Keys grouped by category
MOL_FEATURE_KEYS = (ATOMS, COORDINATES)
TARGET_KEYS = (ENERGY,)
EDGE_INDEX_KEYS = (RECEIVERS, SENDERS)
COUNT_KEYS = (N_ATOMS, N_EDGES)
INDEX_KEYS = (BATCH_INDEX, INDEX, MOLECULE_INDEX)

ALL_KEYS = (
    *MOL_FEATURE_KEYS,
    *TARGET_KEYS,
    *EDGE_INDEX_KEYS,
    *COUNT_KEYS,
    *INDEX_KEYS,
)

# Grouping the keys by unbatching procedure and shape
ATOMIC_KEYS = (
    ATOMS,
    COORDINATES,
    BATCH_INDEX,
)  # (n_atoms,x)
MOLECULAR_KEYS = (
    ENERGY,
    *COUNT_KEYS,
    INDEX,
    MOLECULE_INDEX,
)  # (n_mols, 1)
# Data fields end


# Input to the graph network is a dictionary with ALL_KEYS being the keys
class MolDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def replace(self, **kwargs):
        new_mol_dict = self.copy()  # Shallow copy.
        # The dict is in new memory but the values are the same as the old
        new_mol_dict = MolDict(**new_mol_dict)
        for key, value in kwargs.items():
            new_mol_dict[key] = value
        return new_mol_dict

    def __str__(self):
        return self._print()

    def __repr__(self):
        return self._print()

    def _print(self):
        output = []
        for key, value in self.items():
            if value is None:
                output.append("{}: None".format(key))
            else:
                output.append(
                    "{}: {}, {}".format(key, value.shape, value.dtype)
                )
        return "\n".join(output)


def new_mol_dict():
    return MolDict.fromkeys(ALL_KEYS)


class MolDictList(list):
    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        return "MolDictList: {}".format(self.__len__())

    def __repr__(self):
        return "MolDictList: {}".format(self.__len__())


def batch_mol_dicts(mol_dict_list):
    """Batch a list of mol_dicts in a single batched mol_dict

    Parameters:
    mol_dict_list (list): List of mol_dicts

    Returns:
    mol_dict
    """

    batched_mol_dict = new_mol_dict()
    # To batch the mol_dicts we vstack all the arrays
    for KEY in ALL_KEYS:
        batched_mol_dict[KEY] = np.vstack(
            [mol_dict[KEY] for mol_dict in mol_dict_list]
        )

    # Here we adjust the index keys so they correspond to the
    # Correct atom numbers
    atom_index_start = 0
    edge_index_start = 0
    for i, (n_atoms, n_edges) in enumerate(
        zip(
            batched_mol_dict[N_ATOMS].flatten(),
            batched_mol_dict[N_EDGES].flatten(),
        )
    ):
        atom_index_end = atom_index_start + n_atoms
        edge_index_end = edge_index_start + n_edges
        batched_mol_dict["batch_index"][atom_index_start:atom_index_end] += i

        for key in EDGE_INDEX_KEYS:
            batched_mol_dict[key][
                edge_index_start:edge_index_end
            ] += atom_index_start

        atom_index_start += n_atoms
        edge_index_start += n_edges

    return batched_mol_dict


def unbatch_mols(batched_mol_dict):
    """Return a list of mol_dicts"""

    n_atoms = batched_mol_dict["n_atoms"].flatten()
    n_edges = batched_mol_dict["n_edges"].flatten()
    n_mols = len(n_atoms)
    if n_mols == 1:
        return batched_mol_dict

    mol_dicts = MolDictList(new_mol_dict() for i in range(n_mols))
    atom_index_start = 0
    edge_index_start = 0
    for i, (n_atoms, n_edges) in enumerate(zip(n_atoms, n_edges)):
        atom_index_end = atom_index_start + n_atoms
        edge_index_end = edge_index_start + n_edges

        for KEY in ATOMIC_KEYS:
            # The OPTIONAL_KEYS area subset of ATOMIC_KEYS
            if KEY not in batched_mol_dict:
                continue
            mol_dicts[i][KEY] = batched_mol_dict[KEY][
                atom_index_start:atom_index_end
            ]

        mol_dicts[i][BATCH_INDEX] -= i

        for KEY in EDGE_INDEX_KEYS:
            mol_dicts[i][KEY] = (
                batched_mol_dict[KEY][edge_index_start:edge_index_end]
                - atom_index_start
            )

        for KEY in MOLECULAR_KEYS:
            if KEY not in batched_mol_dict:
                continue
            mol_dicts[i][KEY] = np.array([batched_mol_dict[KEY][i]])

        atom_index_start = atom_index_end
        edge_index_start = edge_index_end

    return mol_dicts


def print_mol_dict(mol_dict):
    for key, value in mol_dict.items():
        print("{}: {}, {}".format(key, value.shape, value.dtype))


def save_pickle(path, item):
    with open(path, "wb") as my_file:
        pickle.dump(item, my_file)


def load_pickle(path):
    with open(path, "rb") as my_file:
        data = pickle.load(my_file)
    return data


def save_dataset(path, mol_dict_list):
    "Save mol_dict_list dataset"
    save_pickle(path, batch_mol_dicts(mol_dict_list))


def load_dataset(path):
    "Load a saved mol_dict_list dataset"
    return unbatch_mols(load_pickle(path))
