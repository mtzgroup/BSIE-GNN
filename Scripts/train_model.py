#!/usr/bin/env python3

import argparse
import os
import sys
from NN import data, datagenerator
from tensorflow.keras import callbacks as cb
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from Models import bsie_gnn


def get_args():
    parser = argparse.ArgumentParser(description="Train NN model")
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Size of mini-batches",
    )
    parser.add_argument(
        "-t",
        "--patience",
        type=int,
        help="Early stopping after 'patience' number of epochs",
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        help="Path to datasets. [training, validation].pickle",
    )
    parser.add_argument(
        "-w",
        "--weights_path",
        type=str,
        help="Path to saved weight to be loaded",
    )
    parser.add_argument(
        "-a",
        "--atomic_weights_path",
        type=str,
        help="Path to the constant atomic weights",
    )
    parser.add_argument(
        "-c",
        "--cutoff_distance",
        type=float,
        default=6.0,
        help="Cutoff distance (A). (Default=6.0)",
    )

    args = parser.parse_args().__dict__
    return args


def load_datasets(params, generators=True):
    "Load datasets and datagenerators"
    datasets = {}
    data_dg = {}
    dataset_path = params["dataset_path"]
    for key, required in [
        ("training", True),
        ("validation", True),
        ("testing", False),
    ]:
        path = os.path.join(dataset_path, key + ".pickle")
        dataset = None
        if not os.path.isfile(path):
            if required:
                raise ValueError(f"Could not find {key} dataset at {path}")
            else:
                print("Could not find {} dataset at {}".format(key, path))
        else:
            dataset = data.load_dataset(path)
            print("Loaded {} dataset of size: {}".format(key, len(dataset)))
            datasets[key] = dataset

    if generators:
        data_dg = get_datagenerators(params, datasets)
        return datasets, data_dg
    else:
        return datasets


def get_datagenerators(params, datasets):
    "Get datagenerators for feeding mini-batches to the NN"
    data_dg = {}
    for key, dataset in datasets.items():
        dg = datagenerator.DataGenerator(
            dataset, batch_size=params["batch_size"]
        )
        data_dg[key] = dg

    return data_dg


def get_callbacks(params):
    checkpoint_dir = os.path.join(".", "Checkpoints")
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    early_stopping = cb.EarlyStopping(
        monitor="val_loss",
        patience=params["patience"],
        verbose=1,
        restore_best_weights=False,
    )
    checkpoints_train = cb.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir,
            "weights_{epoch:04d}_{loss:.4e}_trn.hdf5",
        ),
        save_weights_only=True,
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )
    checkpoints_val = cb.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir,
            "weights_{epoch:04d}_{val_loss:.4e}_val.hdf5",
        ),
        save_weights_only=True,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    csv_logger = cb.CSVLogger(
        os.path.join(".", "history.txt"),
        separator="\t",
        append=False,
    )
    return [early_stopping, checkpoints_train, checkpoints_val, csv_logger]


def main():
    params = get_args()
    model = bsie_gnn.get_model(
        cutoff_distance=params["cutoff_distance"],
        atomic_const_weights_path=params["atomic_weights_path"],
        weights_path=params["weights_path"]
    )
    model.compile(
        run_eagerly=False,
        optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mse",
        metrics=[metrics.MeanAbsoluteError()],
    )

    callbacks = get_callbacks(params)
    datasets, data_dg = load_datasets(params)

    _ = model.fit_generator(
        generator=data_dg["training"],
        validation_data=data_dg["validation"],
        epochs=params["epochs"],
        use_multiprocessing=False,
        callbacks=callbacks,
        verbose=2,
    )
    model.save_weights(
        os.path.join(".", "Checkpoints/train_end_weights.hdf5")
    )


if __name__ == "__main__":
    sys.exit(main())
