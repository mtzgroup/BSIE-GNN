import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Embedding,
    Flatten,
    add,
    multiply,
)
from tensorflow.keras.activations import (
    linear,
    # swish
)


def atom_embedding_block(
    atomic_numbers,
    input_dim=20,  # Integer. Size of the vocabulary, i.e. max_atomic_num + 1.
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
        cutoff_distance=None
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
        eta_initial = n_basis_functions**2 / cutoff_distance**2
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
                "cutoff_distance": self.cutoff_distance
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
        trainable=trainable
    )(edge_attr)

    atom_attr = activation(atom_attr)

    receiver_attr = atom_attr
    receiver_attr = Dense(
        atom_attr_dim,
        activation=activation,
        name="{}_rec_dense".format(block_name),
        trainable=trainable
    )(receiver_attr)

    sender_attr = Broadcast()([atom_attr, senders])

    sender_attr = Dense(
        atom_attr_dim,
        activation=activation,
        name="{}_send_dense".format(block_name),
        trainable=trainable
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
            trainable=trainable
        )

    atom_attr = activation(atom_attr)
    atom_attr = Dense(
        atom_attr_dim,
        activation=activation,
        name="{}_dense".format(block_name),
        trainable=trainable
    )(atom_attr)

    for i in range(n_atomic_residual):
        atom_attr = residual_block(
            atom_attr,
            units=atom_attr_dim,
            activation=activation,
            block_name="{}_A_res_{}".format(block_name, i),
            trainable=trainable
        )

    return atom_attr
