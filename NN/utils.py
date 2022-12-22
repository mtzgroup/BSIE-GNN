import numpy as np
import glob
import os
from NN import data


def split_frames(frames, n_skip_lines=0):
    """Split multiple frames (xyz file content) into a list of single frames

    Parameters:
    frames (str): Content of an xyz file
    n_skip_lines (int): number of additional lines at the end of each frame
        (Terachem vel file)

    Returns:
    frame_list (list): List of frames where each frame is a list of lines

    """

    if isinstance(frames, str):
        lines = frames.splitlines()
    elif isinstance(frames, list):
        lines = frames

    n_atoms = int(lines[0].strip())
    reduced_frame_length = n_atoms + 2
    frame_length = reduced_frame_length + n_skip_lines
    n_lines = len(lines)

    n_frames, remainder = np.divmod(n_lines, frame_length)
    frame_list = []
    for i in range(n_frames):
        # If there is any skip line at the end of the frame,
        # this line is not included
        frame = lines[i * frame_length: (i + 1) * frame_length - n_skip_lines]
        frame_list.append(frame)

    return frame_list


def extract_atoms(frame):
    """Get the atoms from a single frame of an xyz file

    Parameters:
    frame (list of str): List of the lines of a frame from xyz file

    Returns:
    atoms (np.ndarray): Array of the atomic symbols
    """
    assert isinstance(frame, list)
    n_atoms = int(frame[0].strip())
    atoms = []

    for line in frame[2: n_atoms + 2]:
        atoms.append(line.split()[0])

    return np.array(atoms)


def extract_coordinates(frame):
    """Extract coordinates from a single xyz frame

    Parameters:
    frame (list): List of lines from a single frame of an xyz file

    Returns:
    coordinates (np.ndarray)
    """
    coordinate_lines = frame[2:]
    coordinates = [
        list(map(float, line.split()[1:])) for line in coordinate_lines
    ]

    return coordinates


def parse_xyz_frames(frames, n_skip_lines=0):
    """Convert xyz file frames to atoms, coordinates and comments

    Parameters:
    frames (str): Frames of an xyz file. (list of lines)
    comments (bool):  whether to return the comment line or not. Default False
    n_skip_lines (int): number of additional lines at the end of each frame
        (Terachem vel file)

    Returns:
    atoms (np.ndarray)
    coordinates (np.ndarray)
    *comments (list)
    """
    if isinstance(frames, str):
        frames = frames.splitlines()
    frames = split_frames(frames, n_skip_lines=n_skip_lines)

    atoms = extract_atoms(frames[0])
    coordinates = np.array(list(map(extract_coordinates, frames)))


    return atoms, coordinates


def read_xyz_file(filename, n_skip_lines=0):
    """Read xyz file

    Arguments
    filename (str): path to xyz file
    n_skip_lines (int): number of additional lines at the end of each frame


    Returns:
    atoms (np.ndarray)
    coordinates (np.ndarray)
    *comments (list)
    """
    with open(filename, "r") as xyz_file:
        frames = xyz_file.read()
    return parse_xyz_frames(
        frames, n_skip_lines=n_skip_lines
    )


def load_np_txt_files(path):
    np_arrays = {}
    filenames = glob.glob(os.path.join(path, "*.txt"))
    for filename in filenames:
        key = os.path.basename(filename).split(".")[0]
        array = np.loadtxt(filename)
        np_arrays[key] = array
    return np_arrays


def create_batches(dataset, batch_size=1000):
    """Create batches of dataset for evaluation"""
    batches = []
    n_batches = int(np.ceil(len(dataset) / batch_size))
    for i in range(n_batches):
        batch = dataset[i * batch_size: (i + 1) * batch_size]
        batch = data.batch_mol_dicts(batch)
        batches.append(batch)
    return batches

