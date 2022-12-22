import tensorflow as tf
from tensorflow import int32
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Add
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Embedding,
    Flatten,
    add,
    multiply,
)
from tensorflow.keras.activations import linear, swish
import numpy as np
import sys

import argparse
# from models import utils
import os


def atom_embedding_block(
    atomic_numbers,
    input_dim=20,  # Integer. Size of the vocabulary, i.e. max_atomic_num + 1 at minimum.
    output_dim=32,
    input_length=1,
    name=None,
    embeddings_initializer="uniform",
    trainable=True,
    **kwargs
):

    atom_attr = Embedding(
        input_dim,
        output_dim,
        input_length=input_length,
        embeddings_initializer=embeddings_initializer,
        name="{}_atom_embedding".format(name),
        trainable=trainable,
        **kwargs
    )(atomic_numbers)
    atom_attr = Flatten()(atom_attr)
    return atom_attr


class PhysNetEdgeEmbeddingBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_basis_functions=32,
        mu_trainable=False,
        beta_trainable=False,
        cutoff_distance=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(cutoff_distance, float):
            raise ValueError("The cutoff distance must be defined as a float")

        self.n_basis_functions = n_basis_functions
        self.mu_trainable = mu_trainable
        self.beta_trainable = beta_trainable
        self.cutoff_distance = cutoff_distance

        self.pi = np.pi
        mu_initial = np.linspace(
            np.exp(-cutoff_distance),
            1,
            num=n_basis_functions,
            dtype=np.float32,
        ).reshape(1, n_basis_functions)
        beta_initial = np.full(
            (1, n_basis_functions),
            (2 / n_basis_functions * (1 - np.exp(-cutoff_distance))) ** -2,
            dtype=np.float32,
        )
        self.mu = tf.Variable(
            initial_value=mu_initial, trainable=mu_trainable, name="mu"
        )
        self.beta = tf.Variable(
            initial_value=beta_initial, trainable=beta_trainable, name="beta"
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_basis_functions": self.n_basis_functions,
                "mu_trainable": self.mu_trainable,
                "beta_trainable": self.beta_trainable,
                "cutoff_distance": self.cutoff_distance,
            }
        )
        return config

    def call(self, coordinates, receivers, senders):
        receiver_coordinates = tf.gather(coordinates, receivers, axis=0)
        sender_coordinates = tf.gather(coordinates, senders, axis=0)
        r_ij = tf.math.sqrt(
            tf.math.reduce_sum(
                tf.square(receiver_coordinates - sender_coordinates), axis=2
            )
        )
        r_ij = tf.reshape(r_ij, shape=(-1, 1))

        phi = (
            1
            - 6 * (r_ij / self.cutoff_distance) ** 5
            + 15 * (r_ij / self.cutoff_distance) ** 4
            - 10 * (r_ij / self.cutoff_distance) ** 3
        )
        rbf = (
            tf.math.exp(-self.beta * (tf.math.exp(-r_ij) - self.mu) ** 2) * phi
        )
        return rbf


class BehlerEdgeEmbeddingBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_basis_functions=32,
        mu_trainable=False,
        eta_trainable=False,
        cutoff_distance=None,
    ):
        super().__init__()
        if not isinstance(cutoff_distance, float):
            raise ValueError("The cutoff distance must be defined as a float")
        self.n_basis_functions = n_basis_functions
        self.mu_trainable = mu_trainable
        self.eta_trainable = eta_trainable
        self.cutoff_distance = cutoff_distance
        self.pi = np.pi
        mu_initial = np.linspace(
            0, cutoff_distance, num=n_basis_functions, dtype=np.float32
        ).reshape(1, n_basis_functions)

        # eta initial comes from (alpha^2=0.5):
        # \sigma = \frac{\alpha \times r_c}{n_{rbf}} \\
        # \eta = \frac{1}{2\sigma^2} = \frac{n_{rbf}^2}{2 \times \alpha^2 \times r_c^2}
        # = \frac{n_{rbf}^2}{2 \times 0.5 \times r_c^2} = \frac{n_{rbf}^2}{r_c^2}
        eta_initial = n_basis_functions ** 2 / cutoff_distance ** 2
        eta_initial = np.full(
            (1, n_basis_functions), eta_initial, dtype=np.float32
        )
        self.mu = tf.Variable(
            initial_value=mu_initial, trainable=mu_trainable, name="mu"
        )
        self.eta = tf.Variable(
            initial_value=eta_initial, trainable=eta_trainable, name="eta"
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_basis_functions": self.n_basis_functions,
                "mu_trainable": self.mu_trainable,
                "eta_trainable": self.eta_trainable,
                "cutoff_distance": self.cutoff_distance,
            }
        )
        return config

    def call(self, coordinates, receivers, senders):
        receiver_coordinates = tf.gather(coordinates, receivers, axis=0)
        sender_coordinates = tf.gather(coordinates, senders, axis=0)
        r_ij = tf.math.sqrt(
            tf.math.reduce_sum(
                tf.square(receiver_coordinates - sender_coordinates), axis=2
            )
        )
        r_ij = tf.reshape(r_ij, shape=(-1, 1))
        gaussian_rbf = tf.math.exp(
            -tf.square(tf.subtract(r_ij, self.mu)) * self.eta
        )
        cutoff_function = 0.5 * (
            tf.math.cos(self.pi * r_ij / self.cutoff_distance) + 1
        )
        rbf = cutoff_function * gaussian_rbf
        return rbf


class AtomConst:
    """Linear regression block."""

    def __init__(self, weights=None, trainable=False):
        self.linear_layer = tf.keras.layers.Dense(
            1,
            use_bias=False,
            trainable=trainable,
            activation="linear",
            name="Linear_regression",
        )
        if weights is not None:
            self.linear_layer.build(weights.size)
            self.linear_layer.set_weights([weights.reshape(-1, 1)])

    def __call__(self, formula=None):
        return self.linear_layer(formula)


class SegmentSum(Layer):
    """Segment sum to sum over atoms or molecules"""

    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, inputs):
        x, index = inputs
        return tf.math.segment_sum(x, tf.reshape(index, [-1]))


class Broadcast(Layer):
    """Broadcast atom features to edges"""

    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, inputs):
        x, index = inputs
        x = tf.gather(x, tf.reshape(index, [-1]))
        return x


def residual_block(
    x,
    units=None,
    activation=None,
    kernel_initializer=None,
    block_name=None,
    trainable=True,
):
    residual = x
    residual = activation(residual)
    residual = Dense(
        units,
        activation=linear,
        kernel_initializer=kernel_initializer,
        name="{}_dense_0".format(block_name),
        trainable=trainable,
    )(residual)
    residual = activation(residual)
    residual = Dense(
        units,
        activation=linear,
        kernel_initializer=kernel_initializer,
        name="{}_dense_1".format(block_name),
        trainable=trainable,
    )(residual)
    x = add([x, residual])
    return x


def output_block(
    x,
    batch_index=None,
    n_residual_blocks=None,
    units=None,
    activation=None,
    kernel_initializer=None,
    block_name=None,
    trainable=True,
):
    for i in range(n_residual_blocks):
        x = residual_block(
            x,
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            block_name="{}_res_{}".format(block_name, i),
            trainable=trainable,
        )
    x = activation(x)
    x = Dense(
        1,
        activation=linear,
        kernel_initializer="zeros",
        name="{}_dense".format(block_name),
        trainable=True,
    )(x)
    x = SegmentSum()([x, batch_index])
    return x


def interaction_block(
    atom_attr=None,
    edge_attr=None,
    receivers=None,
    senders=None,
    atom_attr_dim=None,
    n_interaction_residual=None,
    n_atomic_residual=None,
    activation=None,
    kernel_initializer=None,
    block_name=None,
    trainable=True,
):

    attention_mask = Dense(
        atom_attr_dim,
        activation=linear,
        use_bias=False,
        kernel_initializer="zeros",
        name="{}_mask".format(block_name),
        trainable=trainable,
    )(edge_attr)

    atom_attr = activation(atom_attr)

    receiver_attr = atom_attr
    receiver_attr = Dense(
        atom_attr_dim,
        activation=activation,
        name="{}_rec_dense".format(block_name),
        trainable=trainable,
    )(receiver_attr)

    sender_attr = Broadcast()([atom_attr, senders])

    sender_attr = Dense(
        atom_attr_dim,
        activation=activation,
        name="{}_send_dense".format(block_name),
        trainable=trainable,
    )(sender_attr)
    sender_attr = multiply([sender_attr, attention_mask])
    sender_attr = SegmentSum()([sender_attr, receivers])

    atom_attr = add([receiver_attr, sender_attr])
    for i in range(n_interaction_residual):
        atom_attr = residual_block(
            atom_attr,
            units=atom_attr_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            block_name="{}_res_{}".format(block_name, i),
            trainable=trainable,
        )

    atom_attr = activation(atom_attr)
    atom_attr = Dense(
        atom_attr_dim,
        activation=activation,
        name="{}_dense".format(block_name),
        trainable=trainable,
    )(atom_attr)

    for i in range(n_atomic_residual):
        atom_attr = residual_block(
            atom_attr,
            units=atom_attr_dim,
            activation=activation,
            block_name="{}_A_res_{}".format(block_name, i),
            trainable=trainable,
        )

    return atom_attr


def get_model(
    atom_attr_dim=32,
    edge_attr_dim=32,
    cutoff_distance=6.0,
    mu_trainable=False,
    eta_trainable=False,
    n_interaction_blocks=2,
    n_interaction_residual=2,
    n_atomic_residual=0,
    n_output_residual=1,
    atomic_weights_path=None,
    weights_path=None,
    **kwargs
):
    kernel_initializer = he_uniform()
    activation = swish

    atoms = Input(shape=(1,), name="atoms")
    coordinates = Input(shape=(3,), name="coordinates")
    receivers = Input(shape=(1,), dtype=int32, name="receivers")
    senders = Input(shape=(1,), dtype=int32, name="senders")
    batch_index = Input(shape=(1,), dtype=int32, name="batch_index")

    if atomic_weights_path is not None:
        atomic_weights = [
            np.loadtxt(atomic_weights_path).reshape(-1, 1)
        ]
    else:
        atomic_weights = [np.zeros(20).reshape(-1, 1)]

    atomic_const = atom_embedding_block(
        atoms,
        input_dim=20,
        output_dim=1,
        weights=atomic_weights,
        trainable=False,
        name="AtomicConst",
    )
    atomic_const = SegmentSum()([atomic_const, batch_index])

    atom_attr = atom_embedding_block(
        atoms, input_dim=20, output_dim=atom_attr_dim, name="0", trainable=True
    )
    edge_attr = BehlerEdgeEmbeddingBlock(
        n_basis_functions=edge_attr_dim,
        eta_trainable=eta_trainable,
        mu_trainable=mu_trainable,
        cutoff_distance=cutoff_distance,
    )(coordinates, receivers, senders)

    for i in range(n_interaction_blocks):
        atom_attr = interaction_block(
            atom_attr=atom_attr,
            edge_attr=edge_attr,
            receivers=receivers,
            senders=senders,
            atom_attr_dim=atom_attr_dim,
            n_interaction_residual=n_interaction_residual,
            n_atomic_residual=n_atomic_residual,
            activation=activation,
            kernel_initializer=kernel_initializer,
            block_name="I{}".format(i + 1),
            trainable=True,
        )

    output = output_block(
        atom_attr,
        batch_index=batch_index,
        n_residual_blocks=n_output_residual,
        units=atom_attr_dim,
        activation=activation,
        kernel_initializer=kernel_initializer,
        block_name="O",
        trainable=True,
    )

    output = Add()([atomic_const, output])

    model = Model(
        inputs=[
            atoms,
            coordinates,
            receivers,
            senders,
            batch_index,
        ],
        outputs=[output],
    )

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def get_parameters():
    description = """Final simplified model"""
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "-a",
        "--path",
        type=str,
        help="Path to training dir",
    )
    parser.add_argument(
        "-b",
        "--params_path",
        type=str,
        help="Path to parameter file in json format",
    )
    parser.add_argument(
        "-c",
        "--atom_attr_dim",
        type=int,
        default=32,
        help="Dimensionality of atomic attributes",
    )
    parser.add_argument(
        "-d",
        "--edge_attr_dim",
        type=int,
        default=32,
        help="Dimensionality of edge attributes",
    )
    parser.add_argument(
        "-e",
        "--cutoff_distance",
        type=float,
        default=6.0,
        help="Cutoff distance (Used in edge-embedding)",
    )
    parser.add_argument(
        "-g",
        "--mu_trainable",
        type=bool,
        default=False,
        help="mu in edge-embedding block",
    )
    parser.add_argument(
        "-i",
        "--eta_trainable",
        type=bool,
        default=False,
        help="eta in edge-embedding block",
    )
    parser.add_argument(
        "-j",
        "--n_interaction_blocks",
        type=int,
        default=2,
        help="Number of interaction blocks",
    )
    parser.add_argument(
        "-k",
        "--n_interaction_residual",
        type=int,
        default=2,
        help="Number of interaction residuals",
    )
    parser.add_argument(
        "-l",
        "--n_atomic_residual",
        type=int,
        default=0,
        help="Number of atomic residuals",
    )
    parser.add_argument(
        "-m",
        "--n_output_residual",
        type=int,
        default=1,
        help="Number of residuals in output block",
    )
    parser.add_argument(
        "-o",
        "--atomic_weights_path",
        type=str,
        required=True,
        help="Weights for linear atomic block",
    )
    parameters = parser.parse_args().__dict__
    parameters["model"] = "model_final_2"
    return parameters


# def main():
#     params = get_parameters()
#     params = utils.load_params(params)
#     utils.save_params(params)

#     model = get_model(**params)

#     model.summary()
#     plot_model(
#         model,
#         to_file=os.path.join(params["path"], "model.png"),
#         show_shapes=True,
#         show_layer_names=True,
#         rankdir="TB",
#         expand_nested=False,
#         dpi=96,
#     )


# if __name__ == "__main__":
#     sys.exit(main())
