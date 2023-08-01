from .util_cvr import *
import numpy as np
from ase.io import read as ase_cif_reader
from ase import Atoms


def crystal_parser(filepath=None, string_input=None, error_ok=False, cif_id=None,
                   add_info=True, **kwargs):
    if string_input is not None:
        import io
        string_input = io.StringIO(string_input)
    filename = filepath
    inp = None
    if filepath is not None:
        inp = filepath
    elif string_input is not None:
        inp = string_input
    if (cif_id is None) and (filepath is not None):
        cif_id = os.path.split(filepath)[-1]
    try:
        with mute_warnings():
            atom = ase_cif_reader(inp, **kwargs)
            if add_info:
                atom.info['filename'] = filename
                atom.info['material_id'] = str(cif_id)
                atom.info['min_nearest_neighbor'] = -1.
    except Exception as e:
        if error_ok:
            atom = {'filename': filename, 'error': str(e)}
        else:
            raise
    return atom


def atoms_find_min_dist(atoms: Atoms = None, positions: np.ndarray = None):
    """
    Find the minimum distance between atoms in a structure
    :param atoms: The ASE Atoms object
    :param positions: The positions of the atoms
    :return: float
    """
    from scipy.spatial import distance

    if isinstance(atoms, np.ndarray):
        positions = atoms
        atoms = None

    if atoms is None:
        dist = distance.cdist(positions, positions)
        return min(dist[dist > 0])

    min_dist = atoms.get_cell().cellpar()[:3].sum()  # Initialize with the maximum possible distance

    m = 2
    m = (m, m, m)
    reps = []

    # Finding different images of the atoms
    for m0 in range(m[0]):
        for m1 in range(m[1]):
            for m2 in range(m[2]):
                reps.append(np.dot((m0, m1, m2), atoms.cell))
    reps = np.array(reps)

    # Find the minimum distance between atoms in a structure with periodic boundary conditions
    # Considering all the possible images of the atoms
    pos = atoms.get_positions()
    for i in range(len(pos)):
        for j in range(i, len(pos)):
            dist = distance.cdist(pos[i] + reps, pos[j] + reps)
            min_dist = min(min_dist, min(dist[dist > 0]))
    min_distances = min_dist

    return min_distances
