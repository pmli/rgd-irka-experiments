import subprocess
from pathlib import Path

import numpy as np
from pymor.models.iosys import LTIModel


def gab08():
    """Return the model from Gugercin/Antoulas/Beattie '08.

    This is order-3 example from:
    S. Gugercin, A. C. Antoulas, C. A. Beattie,
    H2 Model Reduction for Large-Scale Linear Dynamical Systems,
    SIAM J. Matrix Anal. Appl.,
    2008,
    doi: 10.1137/060666123
    """
    A = np.array([[0, 1, 0], [0, 0, 1], [-15 / 32, -17 / 16, -2]])
    B = np.array([[0], [0], [1]])
    C = np.array([[5 / 4, 7 / 4, -1]])
    return LTIModel.from_matrices(A, B, C)


def slicot_model(name):
    """Load SLICOT model.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    m : LTIModel
        Model.
    """
    name_list = [
        'CDplayer',
        'beam',
        'build',
        'eady',
        'fom',
        'heat-cont',
        'iss',
        'pde',
        'random',
        'tline',
    ]

    if name not in name_list:
        raise ValueError(f'Unknown model name ({name})')

    if not Path(name + '.mat').exists():
        _download_slicot_model(name)

    return LTIModel.from_mat_file(name)


def _download_slicot_model(name):
    url = f'https://www.slicot.org/objects/software/shared/bench-data/{name}.zip'
    zip_file_name = f'{name}.zip'
    subprocess.run(['wget', url], check=True)
    subprocess.run(['unzip', zip_file_name], check=True)
    subprocess.run(['rm', zip_file_name], check=True)
