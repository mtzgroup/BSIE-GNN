import numpy as np
import tensorflow as tf
from NN import data


def mol_dict_to_input(mol_dict):
    "Function to unpack mol_dict arrays in order expected by the NN"
    input = [
        mol_dict["atoms"],
        mol_dict["coordinates"],
        mol_dict["receivers"],
        mol_dict["senders"],
        mol_dict["batch_index"],
    ]

    return input


class UnpackMolDict:
    """Class to unpack mol_dict for input to NN.

    If target=True, the target is also returned as it is needed for training"""
    def __init__(self, target=True):
        self.target = target
        if target:
            self.unpack_func = self.input_and_target
        else:
            self.unpack_func = self.input_only

    def input_only(self, mol_dict):
        return mol_dict_to_input(mol_dict)

    def input_and_target(self, mol_dict):
        return (
            mol_dict_to_input(mol_dict),
            mol_dict[data.ENERGY],
        )

    def __call__(self, mol_dict):
        return self.unpack_func(mol_dict)


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        target=True,
        batch_size=32,
        shuffle=True,
        drop_remainder=False,
    ):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))
        self.on_epoch_end()  # Shuffle the indices
        self.drop_remainder = drop_remainder
        self.target = target
        self.unpacker = UnpackMolDict(target=target)

    def __len__(self):
        "Denotes the number of batches per epoch"
        if self.drop_remainder:
            return int(np.floor(len(self.dataset) / self.batch_size))
        else:
            return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indices of the batch
        batch_indices = self.indices[
            index * self.batch_size: (index + 1) * self.batch_size
        ]
        batched_mol_dict = data.batch_mol_dicts(
            [self.dataset[i] for i in batch_indices]
        )
        return self.unpacker(batched_mol_dict)

    def on_epoch_end(self):
        "Shuffle indices after each epoch"
        if self.shuffle:
            np.random.shuffle(self.indices)
