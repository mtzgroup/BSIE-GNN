import numpy as np
import pickle

# import pandas as pd
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt

# from mol_graph_net import features

try:  # We want to be able to import this module without having tf
    from mol_graph_net import data
    import tensorflow as tf
    from mol_graph_net.datagenerator import mol_dict_to_input
except ModuleNotFoundError:
    pass

matplotlib.style.use("medium_plots")


def plot_history(
    history,
    start_epoch=None,
    filename=None,
    keys=["loss", "val_loss", "train_loss_t", "val_loss_t"],
    smoothing=False,
    alpha=0.2,
    uniform_filter=None,
    gaussian_filter=None,
    ylim=None,
):
    pd = dict()
    if isinstance(history, dict):
        pd["epoch"] = np.array(history["epoch"], dtype="int")
    else:
        pd["epoch"] = np.array(history.epoch, dtype="int")
        history = history.history

    exclude_keys = []
    for key in keys:
        if key in history:
            pd[key] = np.array(history[key])
        else:
            exclude_keys.append(key)
    for key in exclude_keys:
        keys.remove(key)

    if start_epoch is None:
        start_epoch = pd["epoch"][len(pd["epoch"]) // 2]

    plot_indices = np.argwhere(pd["epoch"] >= start_epoch).flatten()
    if plot_indices.size != 0:
        pd["epoch"] = pd["epoch"][plot_indices]
        for key in keys:
            pd[key] = pd[key][plot_indices]

    transformed = False

    if smoothing:
        transformed = True
        for key in keys:
            pd[key + "_transformed"] = exp_smoothing(pd[key], alpha=alpha)
    if uniform_filter:
        transformed = True
        assert isinstance(uniform_filter, int)
        for key in keys:
            pd[key + "_transformed"] = uniform_filter1d(
                pd[key], uniform_filter
            )
    if gaussian_filter:
        transformed = True
        assert isinstance(gaussian_filter, int)
        for key in keys:
            pd[key + "_transformed"] = gaussian_filter1d(
                pd[key], gaussian_filter
            )

    max_y = 0
    if transformed:
        for key in keys:
            plt.plot(
                pd["epoch"], pd[key + "_transformed"], label=key, zorder=3
            )
            max_y = max(max_y, max(pd[key + "_transformed"]))
            plt.plot(pd["epoch"], pd[key], c="grey", zorder=0)
        plt.ylim(top=max_y * 1.1)
    else:
        for key in keys:
            plt.plot(pd["epoch"], pd[key], label=key, zorder=0)

    # if smoothing:
    #     max_y = 0
    #     for key in keys:
    #         value = exp_smoothing(pd[key], alpha=alpha)
    #         plt.plot(pd["epoch"], value, label=key, zorder=3)
    #         max_y = max(max_y, max(value))
    #     plt.ylim(0,max_y)
    #     for key in keys:
    #         plt.plot(pd["epoch"], pd[key], c="grey", zorder=0)
    # else:
    #     for key in keys:
    #         plt.plot(pd["epoch"], pd[key], label=key, zorder=0)

    plt.xlabel("epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close()


# We want to execute this function in eager mode because
# we rely on python input and output
def gradient_predict(model, input_data):
    with tf.GradientTape() as grad_tape:
        # Convert the coordinates to a Tensor
        input_data[1] = tf.convert_to_tensor(input_data[1])
        grad_tape.watch(input_data[1])
        energy = model(
            input_data,
            training=False,
        )
    gradient = grad_tape.gradient(energy, input_data[1])
    gradient = tf.convert_to_tensor(gradient)
    return energy.numpy(), gradient.numpy()


def get_new_data_dict(unit="Ha"):
    data_dict = {
        "unit": unit,
        "type": None,
        "prediction": [],
        "target": [],
        "mol_number": [],
        "index": [],
    }
    return data_dict


def evaluate_model(
    model,
    *optional_keys,
    batch_size=1000,
    filename=None,
    legend=True,
    v=True,
    **datasets,
):

    eval_dict_e = {}  # Energy
    eval_dict_g = {}  # Gradient
    for partition, dataset in datasets.items():
        if not (
            isinstance(dataset, list) or isinstance(dataset, data.MolDictList)
        ):
            continue

        n_batches = int(np.ceil(len(dataset) / batch_size))

        data_dict_e = get_new_data_dict()
        data_dict_e["type"] = "energy"
        data_dict_g = get_new_data_dict()
        data_dict_g["type"] = "gradient"
        data_dict_g["atoms"] = []

        for i in range(n_batches):
            batch = data.batch_mol_dicts(
                dataset[i * batch_size: (i + 1) * batch_size]
            )
            energy, gradient = gradient_predict(
                model, mol_dict_to_input(batch, *optional_keys)
            )
            data_dict_e["prediction"].append(energy)
            data_dict_g["prediction"].append(gradient)

            for key in ["mol_number", "index"]:
                data_dict_e[key].append(batch[key])
                data_dict_g[key].append(
                    batch[key][batch["batch_index"].flatten()]
                )

            data_dict_e["target"].append(batch["energy"])
            data_dict_g["target"].append(batch["gradient"])
            data_dict_g["atoms"].append(batch["atoms"])
            print("{}: {}/{}".format(partition, i + 1, n_batches), end="\r")
        print("{}: {}/{}".format(partition, i + 1, n_batches))

        for data_dict in [data_dict_e, data_dict_g]:
            for key, value in data_dict.items():
                if isinstance(value, list):
                    data_dict[key] = np.vstack(value)

        eval_dict_e[partition] = data_dict_e
        eval_dict_g[partition] = data_dict_g

    if v:
        print("Energy error:")
        print_eval_dict(eval_dict_e, kcal=True)
        print("\nGradient error")
        print_eval_dict(eval_dict_g, kcal=True)

        plot_eval_dict(
            eval_dict_e,
            filename=filename,
            showclose=True,
            kcal=True,
            legend=legend,
        )

    return eval_dict_e, eval_dict_g


def plot_eval_dict(
    eval_dict, filename=None, showclose=True, kcal=True, legend=True
):

    #  Ha values
    unit = "Ha"
    min_value = 0
    max_value = -100
    margin = 0.01  # Ha
    if kcal:
        unit = "kcal/mol"
        min_value *= 627.5
        max_value *= 627.5
        margin *= 627.5

    get_metrics(eval_dict, kcal=kcal)
    for partition, data_dict in eval_dict.items():
        plt.scatter(
            data_dict["target"].flatten(),
            data_dict["prediction"].flatten(),
            label=partition,
        )
        min_value = min(
            min_value, data_dict["target"].min(), data_dict["prediction"].min()
        )
        max_value = max(
            max_value, data_dict["target"].max(), data_dict["prediction"].max()
        )

    plt.autoscale(enable=False)
    plt.plot([-10000, 10000], [-10000, 10000], c="r")

    plt.xlim((min_value - margin), (max_value + margin))
    plt.ylim((min_value - margin), (max_value + margin))

    plt.xlabel(r"Target / ${}$".format(unit))
    plt.ylabel(r"Prediction / ${}$".format(unit))
    if legend:
        plt.legend()
    if filename:
        plt.savefig(filename)
    if showclose:
        plt.show()
        plt.close()


def get_metrics(eval_dict, kcal=True):
    for key, data_dict in eval_dict.items():
        if not ("target" in data_dict and "prediction" in data_dict):
            raise ValueError('No "target" and "prediction" in data_dict')
        unit = data_dict["unit"]
        assert unit in ["Ha", "kcal/mol"]
        conversion_factor = 1
        if kcal and unit == "Ha":
            conversion_factor = 627.5  # kcal/mol/Ha
            data_dict["unit"] = "kcal/mol"
        if not kcal and unit == "kcal/mol":
            conversion_factor = 1 / 627.5
            data_dict["unit"] = "Ha"

        target = data_dict["target"] * conversion_factor
        prediction = data_dict["prediction"] * conversion_factor
        data_dict["target"] = target
        data_dict["prediction"] = prediction
        error = prediction - target
        data_dict["error"] = error
        data_dict["RMSE"] = np.sqrt(np.mean(np.square(error)))
        data_dict["MAE"] = np.mean(np.abs(error))
        data_dict["STD of MAE"] = np.std(np.abs(error))
        data_dict["STD"] = np.std(error)
        data_dict["ME"] = np.mean(error)
        data_dict["Max_AE"] = np.max(np.abs(error))
        data_dict["MAT"] = np.mean(np.abs(target))  # Mean absolute target
        data_dict["MAPE"] = (
            np.mean(np.abs((prediction - target) / target)) * 100
        )  # Mean absolute percent error


def print_eval_dict(eval_dict, kcal=True):
    block_length = 12
    get_metrics(eval_dict, kcal=kcal)

    first_col_length = max(map(len, eval_dict.keys()))
    first_col_length = max(len("Metrics"), first_col_length) + 1

    print_formats = {
        "RMSE": "{:.3e}",
        "MAE": "{:.3e}",
        "STD of MAE": "{:.3e}",
        "STD": "{:.3e}",
        "ME": "{:+.3e}",
        "Max_AE": "{:.3e}",
        "MAT": "{:.3e}",
        "MAPE": "{:.3e}",
    }

    lines = ["unit_line", "metric_header"]
    for partition, data_dict in eval_dict.items():
        unit = data_dict["unit"]

        line = ["{}".format(partition).ljust(first_col_length, " ") + "# "]
        for metric, format in print_formats.items():
            if metric in data_dict:
                value = data_dict[metric]
                line.append(format.format(value).ljust(block_length, " "))
            else:
                line.append(" " * block_length)
        if line:
            lines.append("".join(line))

    output_length = max(map(len, lines))
    unit_line = " Metrics ({}) ".format(unit)
    remainder = output_length - len(unit_line)

    filler = "#" * (remainder // 2)
    unit_line = f"{filler}{unit_line}{filler}"
    unit_line += "#" * (output_length - len(unit_line))
    lines[0] = unit_line

    metric_header = ["Metrics".ljust(first_col_length, " ") + "# "]
    for metric in print_formats.keys():
        metric_header.append("{}".format(metric).ljust(block_length, " "))
    lines[1] = "".join(metric_header)

    lines.append("#" * output_length)
    print("\n".join(lines))


def save_dict(path, hist_dict):
    with open(path, "wb") as my_file:
        pickle.dump(hist_dict, my_file)


def load_dict(path):
    with open(path, "rb") as my_file:
        dataset = pickle.load(my_file)
    return dataset


def load_hist_txt(path):
    with open(path, "r") as hist_file:
        keys = hist_file.readline().split()
    values = np.loadtxt(path, skiprows=1, unpack=True)
    hist_dict = dict(zip(keys, values))
    hist_dict["epoch"] = hist_dict["epoch"].astype(int)
    return hist_dict


def exp_smoothing(x, alpha=0.2):
    """Single exponential smoothing of time series data"""
    # if isinstance(x, list):
    #     series = np.array(x)
    assert isinstance(x, np.ndarray)
    s = x.copy()
    for i in range(1, x.shape[0]):
        s[i] = s[i - 1] + alpha * (x[i] - s[i - 1])
    return s


# def get_coor_matrix(eval_dict):
#     # energy_data = pd.read_pickle(
# "//home/sorenh/Projects/Basis_set_extrapolation/TF2_graph_network/Data/Psi4_DF/energy_data.pickle")
#     with open(
#         "/home/sorenh/Projects/Basis_set_extrapolation/TF2_graph_network/Data/Psi4_DF/geometry_data.pickle",
#         "rb",
#     ) as geometry_file:
#         geometry_data = pickle.load(geometry_file)

#     get_formula = features.get_formula()
#     keys = ["error", "energy", "n_atoms", "n_heavy_atoms"] + list(
#         features.atom_to_num_dict.keys()
#     )
#     for partition, data_dict in eval_dict.items():
#         error_df = []
#         for error, energy, index in zip(
#             data_dict["error"].flatten(),
#             data_dict["energy"].flatten(),
#             data_dict["index"].flatten(),
#         ):
#             atoms = geometry_data[index]["atoms"]
#             formula = get_formula(atoms)
#             n_atoms = np.sum(formula)
#             n_heavy_atoms = np.sum(formula[1:])
#             error_df.append(
#                 dict(
#                     zip(
#                         keys,
#                         (
#                             error,
#                             energy,
#                             n_atoms,
#                             n_heavy_atoms,
#                             *tuple(formula),
#                         ),
#                     )
#                 )
#             )
#         error_df = pd.DataFrame(error_df)
#         error_df = error_df.drop(columns="P")
#         data_dict["corr"] = error_df.corr()
