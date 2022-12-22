#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np

from NN import data, create_input, utils


def get_arguments():
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Directory where the dataset will be saved",
    )
    parser.add_argument(
        "-x",
        "--coordinate_path",
        type=str,
        help="Path to coordinates",
    )
    parser.add_argument(
        "-e",
        "--energy_path",
        type=str,
        help="Path to file with energies",
    )
    parser.add_argument(
        "-s",
        "--small_basis_set",
        type=str,
        required=True,
        help="""Specify small basis set. If \"None\", target energy
        will be determined by the large basis alone""",
    )
    parser.add_argument(
        "-l",
        "--large_basis_set",
        type=str,
        required=True,
        help="Specify large basis set",
    )
    parser.add_argument(
        "-c",
        "--cutoff_distance",
        type=float,
        required=True,
        help="Cutoff distance (A) for graph construction",
    )
    parser.add_argument(
        "-p",
        "--partitions",
        type=str,
        nargs="+",
        help="Paths the parition files",
    )
    args = parser.parse_args().__dict__
    return args


def process_args(args):
    if args["small_basis_set"] == "None":
        args["small_basis_set"] = None
    else:
        args["small_basis_set"] = args["small_basis_set"].lower()
    args["large_basis_set"] = args["large_basis_set"].lower()

    print(f"Getting coordinates from: {args['coordinate_path']}")
    print(f"Getting energies from:    {args['energy_path']}")
    print(f"Small basis set:          {args['small_basis_set']}")
    print(f"Large basis set:          {args['large_basis_set']}")
    print(f"Cutoff distance:          {args['cutoff_distance']:.1f} A")

    partitions = {}
    if args["partitions"]:
        print("\nPartitions:")
        for path in args["partitions"]:
            basename = os.path.basename(path).split(".")[0]
            indices = set(np.loadtxt(path, dtype=int))
            partitions[basename] = indices
            print(f"{basename:<15} {len(indices):>5} molecules")

    return partitions


def load_energy_data(energy_path):
    print("\nLoading energy data...")
    fp = open(energy_path, "r")

    header = fp.readline()
    header = [item.strip() for item in header.split(",")]
    basis_sets = header[2:]
    print(f"{basis_sets = }")

    energy_data = {}

    count = 0
    for line in fp:
        line = line.split(",")
        molecule_index = int(line[0])
        geometry_index = int(line[1])

        energies = line[2:]
        geom_energy_data = {}
        for basis_set, energy in zip(basis_sets, energies):
            energy = energy.strip()
            if energy:
                energy = float(energy)
            else:
                energy = None
            geom_energy_data[basis_set] = energy

        if molecule_index not in energy_data:
            energy_data[molecule_index] = {}

        energy_data[molecule_index][geometry_index] = geom_energy_data
        count += 1

    fp.close()

    print(
        f"Loaded energy data for {count} geometries of"
        f" {len(energy_data)} unique molecules"
    )
    return energy_data


def load_coordinate_data(coordinate_path):
    print("\nLoading coordinate data. This might take a while...")
    coordinate_data = {}
    count = 0
    groups = os.listdir(coordinate_path)
    for group in groups:
        group_path = os.path.join(coordinate_path, group)
        molecules = os.listdir(group_path)
        for molecule in molecules:
            molecule_index = int(molecule)
            coordinate_data[molecule_index] = {}
            molecule_path = os.path.join(group_path, molecule)
            geometries = os.listdir(molecule_path)
            for geometry in geometries:
                geometry_index = int(geometry.split(".")[0])
                geom_filename = os.path.join(molecule_path, geometry)
                atoms, coordinates = utils.read_xyz_file(geom_filename)

                coordinate_data[molecule_index][geometry_index] = {
                    "atoms": atoms,
                    "coordinates": coordinates,
                }
                count += 1

    print(
        f"Loaded coordinates for {count} geometries of"
        f" {len(coordinate_data)} unique molecules"
    )
    return coordinate_data


def create_dataset(args, energy_data, coordinate_data):
    create_mol_dict = create_input.CreateMolDict(
        cutoff=args["cutoff_distance"]
    )
    dataset = data.MolDictList()
    unique_index = 0
    molecule_indices = list(
        set(energy_data.keys()) and set(coordinate_data.keys())
    )
    for molecule_index in molecule_indices:
        if molecule_index not in energy_data:
            continue
        mol_energy_data = energy_data[molecule_index]
        mol_coord_data = coordinate_data[molecule_index]

        geometry_indices = list(
            set(mol_energy_data.keys()) and set(mol_coord_data.keys())
        )
        for geometry_index in geometry_indices:
            if geometry_index not in mol_energy_data:
                continue
            atoms = mol_coord_data[geometry_index]["atoms"]
            coordinates = mol_coord_data[geometry_index]["coordinates"]
            energy = mol_energy_data[geometry_index][
                args["large_basis_set"].lower()
            ]
            if energy is None:
                continue
            if args["small_basis_set"] is not None:
                small_basis_energy = mol_energy_data[geometry_index][
                    args["small_basis_set"].lower()
                ]
                if small_basis_energy is None:
                    continue
                energy -= small_basis_energy

            dataset.append(
                create_mol_dict(
                    atoms=atoms,
                    coordinates=coordinates,
                    energy=energy,
                    index=unique_index,
                    molecule_index=molecule_index,
                )
            )
            unique_index += 1
    print("Size of dataset: {}".format(len(dataset)))
    return dataset


def split_dataset(partitions, dataset):
    datasets = {}
    for partition_name, partition_indices in partitions.items():
        datasets[partition_name] = [
            item
            for item in dataset
            if item["molecule_index"].item() in partition_indices
        ]
    return datasets


def main():
    args = get_arguments()
    partitions = process_args(args)
    energy_data = load_energy_data(args["energy_path"])
    coordinate_data = load_coordinate_data(args["coordinate_path"])
    dataset = create_dataset(args, energy_data, coordinate_data)

    if not os.path.isdir(args["output_path"]):
        os.makedirs(args["output_path"])

    if not partitions:
        filename = os.path.join(args["output_path"], "dataset.pickle")
        data.save_dataset(filename, dataset)
        print(f"Saved dataset of size {len(dataset)} at {filename}")
    else:
        datasets = split_dataset(partitions, dataset)
        for partition_name, dataset in datasets.items():
            filename = os.path.join(
                args["output_path"], f"{partition_name}.pickle"
            )
            data.save_dataset(filename, dataset)
            print(f"Saved dataset of size {len(dataset)} at {filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
