#!/usr/bin/env python3

import argparse
import os
import sys
from NN import data, utils, create_input, datagenerator
from Models import bsie_gnn
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_args():
    parser = argparse.ArgumentParser(
            description="Make predictions with triained model"
        )
    parser.add_argument(
        "xyz_files",
        nargs="+",
        type=str,
        help="Path to xyz files to make prediction on",
    )
    parser.add_argument(
        "-w",
        "--model_weights",
        nargs="+",
        type=str,
        help="Paths to saved weights of trained model. If multiple weights are given, the mean ensemble is computed",
        required=True
    )
    parser.add_argument(
        "-c",
        "--cutoff_distance",
        type=float,
        default=6.0,
        help="Cutoff distance (A) for graph construction (default=6.0)",
    )

    args = parser.parse_args().__dict__
    return args


def load_xyz_files(filenames):
    xyz_list = []
    for filename in filenames:
        if not os.path.isfile(filename):
            raise ValueError(f"Could not find {filename}")
        xyz_list.append(utils.read_xyz_file(filename))
    return xyz_list


def create_input_batch(xyz_list, cutoff_distance):
    create_mol_dict = create_input.CreateMolDict(
        cutoff=cutoff_distance
    )
    input_data = data.MolDictList()
    for index, (atoms, coordinates) in enumerate(xyz_list):
        input_data.append(
            create_mol_dict(
                atoms=atoms,
                coordinates=coordinates,
                energy=0.0,
                index=index,
                molecule_index=index,
            )
        )
    input_batch = data.batch_mol_dicts(input_data)
    return input_batch


def main():
    args = get_args()

    # Load model and load weights
    models = []
    for weight_filename in args["model_weights"]:
        models.append(
            bsie_gnn.get_model(weights_path=weight_filename)
        )


    # Create the input
    xyz_list = load_xyz_files(args["xyz_files"])
    input_batch = create_input_batch(xyz_list, args["cutoff_distance"])

    # Make predictions
    predictions = []
    for model in models:
        predictions.append(
            model(
                datagenerator.mol_dict_to_input(input_batch),
                training=False
            ).numpy()
        )

    predictions = np.hstack(predictions)
    predictions = np.mean(predictions, axis=1)  # Compute the mean ensemble prediction

    for prediction in predictions:
        print(f"{prediction}")


if __name__ == "__main__":
    sys.exit(main())
