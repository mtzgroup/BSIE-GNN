"""Defines class CreateMolDict to easily create MolDictObjects
Creates a MolDict object for one molecule at a time

The required input fields are the following:
energy: float
gradient: np.ndarray (n_atoms, 3)
atoms: list of atomic_numbers
coordinates: array of coordinates. (atoms and coordinates must correspond)
index: int (Only for bookkeeping)
mol_index: int (Only for bookkeeping)


"""

import numpy as np
from NN import features, data


class CreateMolDict:
    """
    Create input mol_dict for mol_graph_net
    """

    def __init__(self, cutoff=10):
        self.cutoff = cutoff
        self.all_keys = set(data.ALL_KEYS)
        self.atomic_keys = set(data.ATOMIC_KEYS)
        self.molecular_keys = set(data.MOLECULAR_KEYS)

    def __call__(
        self,
        **fields,
    ):
        """The required edge indexing and count keys are created
        in this function. The remaining required keys need to
        given"""

        mol_dict = data.new_mol_dict()
        n_atoms = len(fields[data.ATOMS])

        # Put the values into the mol dict with the correct shape
        for key, value in fields.items():
            if key not in self.all_keys:
                raise ValueError(f"{key} not in data.ALL_KEYS")
            if key in self.atomic_keys:
                mol_dict[key] = np.array(value).reshape(n_atoms, -1)
            elif key in self.molecular_keys:
                mol_dict[key] = np.array([value]).reshape(1, -1)
            else:
                raise ValueError(
                    "Could not identify the correct shape for: {key}"
                )
        mol_dict[data.ATOMS] = features.atomic_symbols_to_numbers(
            mol_dict[data.ATOMS].flatten()
        )

        # Get the receivers and senders, n_atoms, n_edges, and batch_index
        # These required fields are created here in the function
        receivers, senders = features.get_receivers_senders(
            mol_dict[data.COORDINATES], cutoff=self.cutoff
        )
        mol_dict[data.RECEIVERS] = receivers.reshape(-1, 1).astype(
            np.int32
        )
        mol_dict[data.SENDERS] = senders.reshape(-1, 1).astype(
            np.int32
        )
        mol_dict[data.N_ATOMS] = np.array([n_atoms], dtype=np.int32).reshape(
            1, 1
        )
        mol_dict[data.N_EDGES] = np.array(
            [len(receivers)], dtype=np.int32
        ).reshape(1, 1)
        mol_dict[data.BATCH_INDEX] = np.zeros((n_atoms, 1), dtype=np.int32)

        # Change the INDEX_KEYS to int types
        # for KEY in data.INDEX_KEYS:
        #     if KEY in mol_dict:
        #         mol_dict[KEY] = mol_dict[KEY].astype(np.int32)

        #  Change coordinates and target to float32
        for KEY in (data.COORDINATES, data.TARGET_KEYS):
            if KEY in mol_dict:
                mol_dict[KEY] = mol_dict[KEY].astype(np.float32)

        return mol_dict
