from ase.io import read as ase_cif_reader
from utility.utility_general import *
import itertools
from ase import Atoms


# from utility.utility_crystal import *


def cif_parser(filepath, parser='ase', string_input=False, error_ok=False, use_deque=False, tar=None):
    import io

    inp = filepath
    if not isinstance(filepath, list):
        inp = [filepath]
    if tar is not None:
        inp = [tar.extractfile(mem).read().decode('ascii') for mem in filepath]
        string_input = True
    if string_input:
        # inp = [io.StringIO(f.decode('ascii')) for f in filepath]
        inp = [io.StringIO(f) for f in filepath]
    atom_list = []
    if use_deque:
        from collections import deque
        atom_list = deque()
    for i in inp:
        atom = None
        filename = i if isinstance(i, str) else None
        if parser == 'ase':
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    atom = ase_cif_reader(i, format='cif')
                    atom.info['filename'] = filename
            except Exception as e:
                if error_ok:
                    atom = {'filename': filename, 'error': str(e)}
                    # atom.info['error'] = e
                    # atom_list.append(e)
                else:
                    raise e

        atom_list.append(atom)
    if len(atom_list) == 1:
        atom_list = atom_list[0]
    return atom_list


def crystal_id_2_relative_path(ids, id_type='cod', path=None):
    ids_out = []
    if not isinstance(ids, list):
        ids = [ids]
    if id_type == 'cod':
        for _, i in enumerate(ids):
            i = str(i)
            ids_out.append('/'.join([i[0], i[1:3], i[3:5]]) + f'/{i}.cif')
    if path is not None:
        ids_out = [path + i for i in ids_out]
    return ids_out


def atom_to_sparse_3d_image(input_list, len_limit=70., n_bins=128, return_atom=False, skip_min_atomic_dist=False):
    box_corner_points = np.array([p for p in itertools.product([0, 1], repeat=3)]) * n_bins
    n_bins = np.uint8(n_bins)
    len_limit = np.ones(3) * len_limit

    def single_crystal(single_input):
        def check_box_emptiness_error():
            # To check if any atom can be found beyond the corner points of the box
            for point in box_corner_points:
                # point = box_corner_points[i]
                check = np.array([True] * len(Bind_eli))
                # Going over x,y,z separately
                for j in range(3):
                    if point[j] == 0:
                        check = (Bind_eli[:, j] < point[j]) & check
                    else:
                        check = (Bind_eli[:, j] > point[j]) & check
                if not np.any(check):
                    # No point found beyonds it, so there is an error in the structure
                    return True
            # Check if passed the box limitations
            if np.min(np.max(mat_reps, axis=0), axis=0) < np.amin(len_limit, axis=0):
                return True
            return False

        def find_min_dist_cor():
            min_dist = 0
            for point in box_corner_points:
                dist = min(np.sum((b_ind - point) ** 2, axis=1) ** 0.5)
                min_dist = max(dist, min_dist)
            # Converting the #bins to Angstrom
            min_dist = min_dist / n_bins * len_limit[0]
            return min_dist

        filename = single_input
        # Reading CIF
        atoms_unit = None
        if isinstance(filename, Atoms):
            atoms_unit = filename
            filename = atoms_unit.info.get('filename', '')
        if atoms_unit is None:
            atoms_unit = cif_parser(filename)

        if not ('min_nearest_neighbor' in atoms_unit.info):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not skip_min_atomic_dist:
                    atoms_unit.info['min_nearest_neighbor'] = atoms_find_min_dist(atoms_unit)
                else:
                    atoms_unit.info['min_nearest_neighbor'] = 'skipped'
        if not skip_min_atomic_dist:
            if atoms_unit.info['min_nearest_neighbor'] < 0.5:
                raise ValueError('The min nearest neighbor is less than 0.5 A')

        # Calc. atom geometry
        a_unit = atoms_unit.get_positions()
        a_unit -= np.amin(a_unit, axis=0)  # bringing the minimum to the origin

        cell = atoms_unit.get_cell()
        speed = np.sum(cell ** 2, axis=1) ** .5

        if (speed == 0).any():
            raise ValueError('Growth in a direction is zero. (One of the unit vectors is zero!)')

        reps = np.ceil(len_limit / speed + 3).astype('uint8')

        b_types = b_ind = atoms_reps = padding_lost = atoms2 = padding_ind_eli = mat_reps = None
        # for i in range(int(len_limit[0]) // 3):
        while True:
            atoms_reps = atoms_unit.repeat(reps.astype('int'))
            mat_reps = atoms_reps.get_positions()
            mat_reps = mat_reps - np.amin(mat_reps, axis=0)
            mat_reps = mat_reps - (
                    (np.amax(mat_reps, axis=0) - np.amin(mat_reps, axis=0)) / 2 - len_limit / 2)

            b_types = atoms_reps.get_atomic_numbers()
            if np.any(b_types < 1):
                raise Exception('There is negative element in the atomic types.')  # Because there is a negative element
            if np.any(b_types > 118):
                raise Exception('There is element in the atomic types greater than 118')

            # b_ind = np.round(mat_reps / len_limit * n_bins)
            b_ind = np.floor(mat_reps / len_limit * n_bins)
            atoms2 = atoms_reps.copy()
            atoms2.set_positions(b_ind)

            # PADDING
            upper = (b_ind >= np.array((n_bins, n_bins, n_bins))).any(axis=1)
            lower = (b_ind < np.array((0, 0, 0))).any(axis=1)
            eliminations = np.logical_or(upper, lower)
            keep = np.logical_not(eliminations)
            padding_lost = b_types[eliminations]
            padding_ind_eli = np.arange(len(eliminations))[eliminations]
            # B[Bind[keep][:, 0], Bind[keep][:, 1], Bind[keep][:, 2]] = b_types[keep]

            Bind_eli = b_ind[eliminations]
            Btypes_eli = b_types[eliminations]
            b_ind = b_ind[keep]
            b_types = b_types[keep]

            if check_box_emptiness_error():
                reps = reps + 4
                continue
            break

        # check if the box is empty
        if check_box_emptiness_error():
            raise ValueError('There is not enough replications')

        if np.min(np.max(mat_reps, axis=0), axis=0) < np.amin(len_limit, axis=0):
            raise ValueError("Replication error. Not big enough")
        if np.max(np.min(mat_reps, axis=0), axis=0) > 0:
            raise ValueError("Replication error. Axises not properly set to zero")

        unique, counts = np.unique(atoms_reps.get_atomic_numbers(), return_counts=True)
        unique, counts2 = np.unique(np.concatenate((b_types, np.array(padding_lost)), axis=0), return_counts=True)
        if not np.array_equal(counts, counts2):
            raise ValueError("Error in padding.")

        del atoms2[[atom for atom in padding_ind_eli]]
        # a_out = atoms2.get_positions()
        unique, counts = np.unique(atoms2.get_atomic_numbers(), return_counts=True)
        unique, counts2 = np.unique(b_types, return_counts=True)
        if not np.array_equal(counts, counts2):
            raise ValueError("Error in padding.")

        min_dist_corner = find_min_dist_cor().astype('float16')

        if np.any(b_types < 1):
            raise ValueError('There is a negative element in the types of atoms')

        b_types8 = b_types.astype('uint8')
        bind8 = b_ind.astype('uint8')
        if not np.all(b_types == b_types8) or not np.all(bind8 == b_ind):
            raise Exception('There is a mistake in the type conversion. MUST BE TAKEN CARE OF!')

        if return_atom:
            return atoms2
        return {'reps': reps, 'Binds': bind8, 'n_bins': n_bins,
                'Btypes': b_types8, 'filename': filename,
                'min_dist_corner': min_dist_corner}

    if not isinstance(input_list, list):
        input_list = [input_list]

    out = []
    for inp in input_list:
        try:
            out.append(single_crystal(inp))
        except Exception as e:
            out.append({'input': inp, 'error': str(e)})
    if len(out) == 1:
        out = out[0]
    return out


def expand_spars_3d_image(data, n_bin=None, channels_list=None):
    if channels_list is None:
        channels_list = ['atomic_number', 'group', 'period']
    if not isinstance(data, list):
        data = [data]
    if n_bin is None:
        # print(data)
        n_bin = data[0]['n_bins']
    mat2 = np.zeros((len(data), n_bin, n_bin, n_bin, len(channels_list)), dtype=np.uint8)
    for i in range(len(data)):
        bind = data[i]['Binds']
        channels = channel_generator(data[i]['Btypes'], channels_list)
        for c in range(channels.shape[1]):
            mat2[i, bind[:, 0], bind[:, 1], bind[:, 2], c] = channels[:, c]
    return mat2


def channel_generator(b_types, channels_list=None):
    if channels_list is None:
        channels_list = ['atomic_number', 'group', 'period']
    # b_types[b_types < 1] = 1
    # if np.any(b_types < 1):
    #     raise ValueError('There is a negative element in the types of atoms')

    # periodic_table = pt.elements
    from utility.utility_crystal import PeriodicTable
    periodic_table = PeriodicTable().table
    # end of temp changes
    try:
        channels_c_g = np.zeros((b_types.shape[0], len(channels_list)), dtype=np.uint8)
        for c in range(len(channels_list)):
            if channels_list[c] == 'atomic_number':
                for i in range(len(b_types)):
                    channels_c_g[i, c] = b_types[i]
            if channels_list[c] == 'group':
                for i in range(len(b_types)):
                    g = 0
                    if periodic_table[channels_list[c]][b_types[i] - 1] == '-':
                        if (b_types[i] >= 57) & (b_types[i] <= 71):
                            g = b_types[i] - 57 + 1 + 18
                        if (b_types[i] >= 89) & (b_types[i] <= 103):
                            g = b_types[i] - 89 + 1 + 18
                    else:
                        g = int(periodic_table[channels_list[c]][b_types[i] - 1])
                    if g == 0:
                        raise ValueError('g can not be zero')
                    channels_c_g[i, c] = g
            if channels_list[c] == 'period':
                for i in range(len(b_types)):
                    channels_c_g[i, c] = int(periodic_table[channels_list[c]][b_types[i] - 1])
    except Exception as e:
        save_var(locals(), 'sessions/channel_generator.pkl')
        print('Ali: Session saved at sessions/ChannelGenerator.pkl')
        print(e)
        raise
    return channels_c_g


def atoms_find_min_dist(atoms_1, return_dict=False, ok_error=False):
    from scipy.spatial import distance
    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList

    def natural_cutoffs(atoms, mul=1, **kwargs):
        return [kwargs.get(atom.symbol, covalent_radii[atom.number] * mul)
                for atom in atoms]

    def build_neighbor_list(atoms, cutoffs=None, **kwargs):
        if cutoffs is None:
            cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, **kwargs)
        nl.update(atoms)
        return nl

    if isinstance(atoms_1, dict) & ok_error:
        return {'min_atomic_dist': 0,
                'filename': atoms_1['filename']}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if atoms_1.get_number_of_atoms() == 1:
            # if atoms_1.get_global_number_of_atoms() == 1:
            min_dist = min(np.linalg.norm(atoms_1.cell, axis=1))

        elif atoms_1.get_number_of_atoms() < 250:
            # elif atoms_1.get_global_number_of_atoms() < 250:
            pos = atoms_1.get_positions()
            dist = distance.cdist(pos, pos)
            min_dist = min(dist[dist > 0])
        else:
            neighbor_list = build_neighbor_list(atoms_1)
            pos = atoms_1.get_positions()
            min_dist = min(np.linalg.norm(atoms_1.cell, axis=1))
            for k in neighbor_list.nl.neighbors:
                dist = distance.cdist(pos[k], pos[k])
                if len(dist[dist > 0]) < 2:
                    continue
                min_dist = min(min_dist, min(dist[dist > 0]))
    if return_dict:
        return {'min_atomic_dist': min(min_dist, min(np.linalg.norm(atoms_1.cell, axis=1))),
                'filename': atoms_1.info['filename']}
    return min(min_dist, min(np.linalg.norm(atoms_1.cell, axis=1)))


def cif2image(atom, save_to=None, compressed_save=False, create_path=False, ok_error=False,
              skip_min_atomic_dist=False):
    if isinstance(atom, str):
        atom = cif_parser(atom)
    try:
        sparse_img = atom_to_sparse_3d_image(atom, skip_min_atomic_dist=skip_min_atomic_dist)
        img = expand_spars_3d_image(sparse_img)
    except Exception as e:
        if ok_error:
            pass
            return False
        else:
            raise e
    if save_to is not None:
        if create_path:
            os.makedirs('/'.join(save_to.split('/')[:-1]), exist_ok=True)
        if compressed_save:
            np.savez_compressed(save_to, img)  # *.npy
        else:
            np.save(save_to, img)  # *.npz
        return True
    return img


def data_set_img_2_h5(df, filename, img_per_chunk=50):
    import h5py
    import tqdm
    fz = h5py.File(filename, 'w')

    # img_per_chunk = 50
    arr = np.arange(len(df))
    arr = np.array_split(arr, (len(arr) - 1) // img_per_chunk + 1)

    # img_list = []
    for i in tqdm.tqdm_notebook(range(len(arr))):
        inds = arr[i]
        #     img_path = df_train['img_path'].iloc[i] + '.npy'
        #     img_list += [np.load(img_path)]
        img_list = [np.load(df['img_path'].iloc[j] + '.npy') for j in inds]
        img_list = np.concatenate(img_list)

        labels = df['y'].iloc[inds]
        filenames = df['img_path'].iloc[inds]
        asciiList = [n.encode("ascii", "ignore") for n in filenames]

        fz.create_dataset(f'images_{i}', data=img_list, dtype='uint8')
        fz.create_dataset(f'labels_{i}', data=labels, dtype='uint8')
        fz.create_dataset(f'filenames_{i}', (len(asciiList), 1), 'S100', asciiList, )

    fz.close()


if __name__ == '__main__':
    # cif_files = list_all_files(f'{old_data_path}cod/cif', pattern='**/*.cif', shuffle=True)

    # atoms = cif_parser(cif_files[:5], error_ok=True)

    # df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
    #                    'num_wings': [2, 0, 0, 0],
    #                    'num_specimen_seen': [10, 2, 1, 8]},
    #                   index=['falcon', 'dog', 'spider', 'fish'])
    #
    # for i in df['num_legs']:
    #     print(i)

    # a = atom_to_sparse_3d_image(cif_files[:30])
    # err = [a.pop(i) for i in reversed(list(range(len(a)))) if 'error' in a[i]]
    # b = expand_spars_3d_image(a)
    # atom_to_sparse_3d_image('/home/ali/Data/cod/cif/4/30/43/4304374.cif')
    # create_3d_images_datasets('/home/ali/Data/cod/cif/4/30/43/4304374.cif', 5)

    cif2image('/home/ali/Data/cod/cif/4/30/43/4304374.cif', skip_min_atomic_dist=True)
    # cif2image('/home/ali/Data/cod/cif/2/20/96/2209646.cif')
    print('End')
