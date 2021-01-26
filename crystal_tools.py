from utility.util_crystal import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from ase.io import read, write
import statistics
from functools import reduce
from math import gcd
from ase import Atoms


def atom2mat(filename, len_limit=50., n_bins=64, atoms_unit=None, return_atom=False):
    """
    Converting the ASE Atom Objects (Crystal structures) to sparse 3D images.

    :param filename:
    :param len_limit:
    :param n_bins:
    :param atoms_unit:
    :param return_atom:
    :return:
    """
    points_to_check = np.asarray([[0, 0, 0], [0, 0, n_bins], [0, n_bins, 0], [n_bins, 0, 0],
                                  [n_bins, n_bins, 0], [n_bins, 0, n_bins], [0, n_bins, n_bins],
                                  [n_bins, n_bins, n_bins]])

    def check_box_emptiness_error():
        """
        Finds if some atoms passes the corner of the created cube before cutting. If return False, the unit cell
        has been replicated enough to fill the cube
        :return:
        """
        # To check if any atom can be found beyond the corner points of the box
        for i in range(len(points_to_check)):
            point = points_to_check[i]
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
        """
        Finds the maximum of the minimum-distance of the super cell atoms from the corners of the cube
        :return:
        """
        min_dist = 0
        for i in range(len(points_to_check)):
            dist = min(np.sum((Bind - points_to_check[i]) ** 2, axis=1) ** 0.5)
            min_dist = max(dist, min_dist)
        # Converting the #bins to Angstrom
        min_dist = min_dist / n_bins * len_limit[0]
        return min_dist

    n_bins = np.uint8(n_bins)
    # Reading CIF
    if isinstance(filename, Atoms):
        atoms_unit = filename
        filename = ''
    if atoms_unit is None:
        if not exists(filename):
            raise FileNotFoundError('File were not found')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                atoms_unit = cif_parser(filename)
            except Exception as e:
                raise Exception(f'ASE can\'t  read the file. {e}')
    else:
        if hasattr(atoms_unit, 'filename'):
            filename = atoms_unit.filename
        else:
            if not hasattr(atoms_unit, 'filename_full'):
                atoms_unit.filename_full = None
                atoms_unit.filename = None
            filename = atoms_unit.filename_full
    if not hasattr(atoms_unit, 'min_nearest_neighbor'):
        atoms_unit.min_nearest_neighbor = find_min_dist(atoms_unit)
    if atoms_unit.min_nearest_neighbor < 0.5:
        raise ValueError('The min nearest neighbor is less than 0.5 A')
    # Calc. atom geometry
    a_unit = atoms_unit.get_positions()
    a_unit -= np.amin(a_unit, axis=0)  # bringing the minimum to the origin

    len_limit = np.ones((3)) * len_limit
    cell = atoms_unit.get_cell()
    speed = np.sum(cell ** 2, axis=1) ** .5
    if (speed == 0).any():
        raise ValueError('Growth in a direction is zero')
    reps = np.ceil(len_limit / speed + 3).astype('uint8')

    Btypes = Bind = atoms_reps = padding_lost = atoms2 = padding_ind_eli = mat_reps = None
    for i in range(int(len_limit[0]) // 3):
        atoms_reps = atoms_unit.repeat(reps.astype('int'))
        mat_reps = atoms_reps.get_positions()
        mat_reps = mat_reps - np.amin(mat_reps, axis=0)
        mat_reps = mat_reps - (
                (np.amax(mat_reps, axis=0) - np.amin(mat_reps, axis=0)) / 2 - len_limit / 2)

        Btypes = atoms_reps.get_atomic_numbers()
        if np.any(Btypes < 1):
            raise Exception('There is negative element in the atomic types.')  # Because there is a negative element
        if np.any(Btypes > 118):
            raise Exception('There is element in the atomic types greater than 118')

        Bind = np.round(mat_reps / len_limit * n_bins)
        atoms2 = atoms_reps.copy()
        atoms2.set_positions(Bind)

        # PADDING
        upper = (Bind >= np.array((n_bins, n_bins, n_bins))).any(axis=1)
        lower = (Bind < np.array((0, 0, 0))).any(axis=1)
        eliminations = np.logical_or(upper, lower)
        keep = np.logical_not(eliminations)
        padding_lost = Btypes[eliminations]
        padding_ind_eli = np.arange(len(eliminations))[eliminations]
        # B[Bind[keep][:, 0], Bind[keep][:, 1], Bind[keep][:, 2]] = Btypes[keep]
        Bind_eli = Bind[eliminations]
        Btypes_eli = Btypes[eliminations]
        Bind = Bind[keep]
        Btypes = Btypes[keep]

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
    # unique, counts2 = np.unique(np.concatenate((B[B != 0], np.array(padding_lost)), axis=0), return_counts=True)
    unique, counts2 = np.unique(np.concatenate((Btypes, np.array(padding_lost)), axis=0), return_counts=True)
    if not np.array_equal(counts, counts2):
        raise ValueError("Error in padding.")

    del atoms2[[atom for atom in padding_ind_eli]]
    a_out = atoms2.get_positions()
    unique, counts = np.unique(atoms2.get_atomic_numbers(), return_counts=True)
    unique, counts2 = np.unique(Btypes, return_counts=True)
    if not np.array_equal(counts, counts2):
        raise ValueError("Error in padding.")

    min_dist_corner = find_min_dist_cor().astype('float16')

    btypes8 = Btypes.astype('uint8')
    bind8 = Bind.astype('uint8')
    if not np.all(Btypes == btypes8) or not np.all(bind8 == Bind):
        raise Exception('There is a mistake in the type conversion. MUST BE TAKEN CARE OF!')

    # 'min': np.amin(a_out, axis=0), 'max': np.amax(a_out, axis=0)
    # 'symbols': atoms_unit.symbols
    if return_atom:
        return atoms2
    return {'reps': reps, 'Binds': bind8,
            'Btypes': btypes8, 'filename': filename,
            'min_dist_corner': min_dist_corner}


def spars_mat_2_3d_mat(data, n_bin, channels_list):
    """
    Expands a sparse 3D matrix

    :param data:
    :param n_bin:
    :param channels_list:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    mat2 = np.zeros((len(data), n_bin, n_bin, n_bin, len(channels_list)))
    for i in range(len(data)):
        # print(f'i = {i}')
        bind = data[i]['Binds']
        channels = channel_generator(data[i]['Btypes'], channels_list)
        for c in range(channels.shape[1]):
            mat2[i, bind[:, 0], bind[:, 1], bind[:, 2], c] = channels[:, c]
    return mat2


def channel_generator(b_types, chan_list):
    """
    Assigning the channels of the image based on the atomic numbers
    :param b_types:
    :param chan_list:
    :return:
    """
    b_types[b_types < 1] = 1
    if np.any(b_types < 1):
        raise ValueError('There is a negative element in the types of atoms')

    periodic_table = pt.elements
    # end of temp changes
    try:
        channels_c_g = np.zeros((b_types.shape[0], len(chan_list)))
        for c in range(len(chan_list)):
            for i in range(len(b_types)):
                # print(f'i = {i}')
                if chan_list[c] == 'atomic_number':
                    channels_c_g[i, c] = b_types[i] / 118
                if chan_list[c] == 'group':
                    if periodic_table[chan_list[c]][b_types[i] - 1] == '-':
                        g = 3.5
                    else:
                        g = int(periodic_table[chan_list[c]][b_types[i] - 1])
                    channels_c_g[i, c] = g / 19
                if chan_list[c] == 'period':
                    channels_c_g[i, c] = int(periodic_table[chan_list[c]][b_types[i] - 1]) / 8
    except Exception as e:
        save_var(locals(), 'sessions/channel_generator.pkl')
        print('Ali: Session saved at sessions/ChannelGenerator.pkl')
        print(e)
        raise
    return channels_c_g


def plot_lattice_parameters(data_set='cod/data_sets/all/cif_chunks', return_data=False):
    if exists('tmp/lattice_parameters.pkl'):
        lattice_parameters = load_var('tmp/lattice_parameters.pkl')
    elif data_set is None:
        lattice_parameters = run_query("select a, b, c, nel from cod_data", make_table=True)
    else:
        chunks = list_all_files(data_path + data_set)
        print('Creating the min list, {} files found.'.format(len(chunks)))
        min_list = []
        for i in range(len(chunks)):
            print('Analyzing: ', chunks[i], '\t Memory = ', memory_info())
            atoms_list = load_var(chunks[i])
            for j in range(len(atoms_list)):
                # if j in np.floor(np.linspace(0, len(atoms_list) - 1, 5)):
                #     print('{} out of {} samples. Memory = {}'.format(j, len(atoms_list), memory_info()))
                atoms_py_mat = atoms_list[j]
                if isinstance(atoms_py_mat, dict):
                    atoms_py_mat = atoms_py_mat['ase']
                    if isinstance(atoms_py_mat, str):
                        continue
                cell = np.sum(atoms_py_mat.get_cell() ** 2, axis=1) ** .5
                min_list.append({'a': cell[0], 'b': cell[1], 'c': cell[2]})
        lattice_parameters = pandas.DataFrame(min_list)
    # where should be removed ........
    lattice_parameters['min_lattice'] = lattice_parameters[['a', 'b', 'c']].min(axis=1)
    lattice_parameters['max_lattice'] = lattice_parameters[['a', 'b', 'c']].max(axis=1)
    for i in ['min_lattice', 'max_lattice']:
        cond = ~lattice_parameters.applymap(np.isreal)[i]
        lattice_parameters = lattice_parameters[~cond]
        lattice_parameters.reset_index(drop=True, inplace=True)
    save_var(lattice_parameters, 'tmp/lattice_parameters.pkl', make_path=True)

    if return_data:
        return lattice_parameters

    f, ax = plt.subplots(figsize=(92 / 100 * 3.93701, 70 / 100 * 3.93701))
    font_size = 6
    sns.set(style="darkgrid", rc={"lines.linewidth": 1}, color_codes=True)
    bins = 200
    sns.distplot(lattice_parameters['min_lattice'], bins=bins, ax=ax)
    sns.distplot(lattice_parameters['max_lattice'], bins=bins, ax=ax)
    axes = plt.gca()
    axes.set_xlim([0, 35])
    makedirs('plots/', exist_ok=True)
    for i in ['a', 'b', 'c']:
        sns.distplot(lattice_parameters[i], bins=bins, ax=ax)
    plt.legend(['$Min. of Lattice Constants$', '$Max. of Lattice Constants$', '$a$', '$b$', '$c$'],
               fontsize=font_size)
    axes.set_xlim([0, 35])
    ax.tick_params(labelsize=font_size)
    plt.xlabel("$Lattice Constants (\AA)$", fontsize=font_size)
    plt.ylabel('$Probability Distribution Function$', fontsize=font_size)
    plt.title('$Lattice Parameters Distribution$', fontsize=font_size)
    plt.savefig('plots/Lattice Parameters Distribution2.svg')
    plt.show()


def plot_nearest_neighbors(data_set='cod/data_sets/all_pymatgen_5e-2/cif_chunks', exclude=None):
    filename = 'tmp/nearest_neighbors_____.pkl'
    if exists(filename):
        print('Loading the min list')
        min_list = load_var(filename)
    else:
        chunks = list_all_files(data_path + data_set)
        print('Creating the min list, {} files found.'.format(len(chunks)))
        min_list = []
        for i in range(len(chunks)):
            print('Analyzing: ', chunks[i], '\t Memory = ', memory_info())
            atoms_list = load_var(chunks[i])
            for j in range(len(atoms_list)):
                atoms_py_mat = atoms_list[j]
                path = atoms_py_mat.filename
                if len(re.findall('super', path)) > 0:
                    continue
                if exclude is not None:
                    if not isinstance(exclude, list):
                        exclude = [exclude]
                    if not include_acceptable_elements(atoms_py_mat, exclusions=exclude):
                        continue
                if hasattr(atoms_py_mat, 'min_nearest_neighbor'):
                    min_list.append(atoms_py_mat.min_nearest_neighbor)
                else:
                    try:
                        min_list.append(find_min_dist(atoms_py_mat))
                    except Exception as e:
                        print(Color.BOLD + Color.RED + f'{path}, Error: {e}' + Color.END)
        min_list = np.asarray(min_list)
        save_var(min_list, filename, make_path=True)
    sns.set(style="ticks", rc={"lines.linewidth": 2}, color_codes=True)
    bins = 200
    sns.distplot(min_list, bins=bins, kde_kws={"lw": 1.5, "label": "KDE"})
    axes = plt.gca()
    axes.set_xlim([0, 4])
    plt.xlabel('Min of Nearest Neighbors (A)')
    plt.ylabel('PDF')
    plt.title('Nearest Neighbors Distribution')
    plt.legend(['KDE approximation', 'Bins'])
    makedirs('plots/', exist_ok=True)
    plt.savefig('plots/Nearest Neighbors Distribution.svg')
    plt.show()


def find_min_dist(atoms_1):
    """
    Finds the minimum distance of atoms in a crystal structure
    :param atoms_1: ASE Atoms Object
    :return:
    """
    from ase.neighborlist import NeighborList
    from ase.data import covalent_radii

    def natural_cutoffs(atoms, mult=1, **kwargs):
        return [kwargs.get(atom.symbol, covalent_radii[atom.number] * mult)
                for atom in atoms]

    def build_neighbor_list(atoms, cutoffs=None, **kwargs):
        if cutoffs is None:
            cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, **kwargs)
        nl.update(atoms)
        return nl

    if atoms_1.get_number_of_atoms() == 1:
        min_dist = min(np.linalg.norm(atoms_1.cell, axis=1))

    elif atoms_1.get_number_of_atoms() < 250:
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
    return min(min_dist, min(np.linalg.norm(atoms_1.cell, axis=1)))


def cif_parser(filename, site_tolerance=1e-3, parser='ase', check_min_dist_cond=False,
               input_format: str = None, check_size_of_crystal_cond=False):
    """
    Reads a CIF structure from a file or from string

    :param filename:
    :param site_tolerance:
    :param parser:
    :param check_min_dist_cond:
    :param input_format:
    :param check_size_of_crystal_cond:
    :return:
    """
    atom = None
    # if not isinstance(site_tolerance, str):
    if not parser == 'ase':
        try:
            from pymatgen.io import cif
            from pymatgen.io import ase as pymatgen_ase

            A = cif.CifParser(filename, site_tolerance=site_tolerance)
            atom = pymatgen_ase.AseAtomsAdaptor.get_atoms(A.get_structures()[0])
        except Exception as e:
            raise e
    if site_tolerance == 'ase' or parser == 'ase':
        try:
            atom = read(filename, format=input_format)
        except Exception as e:
            raise e
    atom.min_nearest_neighbor = find_min_dist(atom)
    atom.filename_full = filename
    atom.filename = filename
    atom.site_tolerance = site_tolerance if not isinstance(site_tolerance, str) else 0
    # atom.parser = 'pymatgen' if not isinstance(site_tolerance, str) else 'ase'
    atom.parser = parser
    if len(re.findall('super', filename.split('/')[-1])) > 0:
        atom.super_cell = True
    else:
        atom.super_cell = False
    if check_min_dist_cond:
        if atom.min_nearest_neighbor < 0.5:
            raise ValueError('Atoms are sitting too close to each other, model resolution error.')

    if check_size_of_crystal_cond:
        cell = np.sum(atom.get_cell() ** 2, axis=1) ** .5
        if max(cell) > 35:
            raise ValueError('The crystal structure is too large and can not be replicated enough.')
    return atom


def atom2prop(atom, **kwargs):
    """
    The helper function to 'distribution' function for calculating the attributes.
    :param atom: ASE Atom Object
    :param kwargs:
    :return:
    """
    prop = kwargs['prop']
    if isinstance(atom, dict):
        if prop == 'density-mat':
            atom['density-mat'] = PeriodicTable().table['atomic mass'][atom['Btypes'] - 1].sum() / 70 ** 3
        prop = atom[prop]
        filename = atom['filename']
    else:
        if prop == 'density':
            atom.density = sum(atom.get_masses()) / atom.get_volume()
        if prop == 'spacegroup':
            atom.spacegroup = atom.info['spacegroup'].no
        if prop == 'atomic_number':
            (unique, counts) = np.unique(atom.get_atomic_numbers(), return_counts=True)
            g = reduce(gcd, counts)
            counts = counts / g
            atom.prop = []
            for i in range((len(counts))):
                atom.prop += [unique[i]] * int(counts[i])
            if max(atom.prop) > 118 or min(atom.prop) < 1:
                red_print('stop')
                raise Exception('Atomic numbers error')
            return atom.prop
        if prop == 'symbols':
            return set(atom.symbols)
        prop = getattr(atom, prop)
        filename = atom.filename if hasattr(atom, 'filename') else atom.filename_full
    if prop > 25:
        # red_print(f'{filename} , Prop: {prop}')
        prop
    return prop


def distribution(data_sets=None, property_fn=atom2prop, names=None, previous_run=None, bar=False,
                 bins=None, x_lim_lo=-100., x_lim_hi=100., equal_axis=False, normalization=False,
                 threshold_z_score=6, threshold_iqr=99, outlier_detection=True, return_data=False,
                 exclude_name=None, exclude_condition=None,
                 **kwargs):
    """
    Computes different distribution of properties in a dataset

    :param data_sets:
    :param property_fn:
    :param names:
    :param previous_run:
    :param bar:
    :param bins:
    :param x_lim_lo:
    :param x_lim_hi:
    :param equal_axis:
    :param normalization:
    :param threshold_z_score:
    :param threshold_iqr:
    :param outlier_detection:
    :param return_data:
    :param exclude_name:
    :param exclude_condition:
    :param kwargs:
    :return:
    """
    if exclude_name is None:
        exclude_name = ''
    if isinstance(exclude_condition, str):
        exclude_condition = [exclude_condition]
    if names is None:
        names = data_sets
    if bins is None:
        bins = [40] * len(data_sets)
    if not isinstance(bins, list):
        bins = [bins] * len(data_sets)
    data = {}
    x_max = 0
    prop = kwargs['prop']
    for n in names:
        data.update({n: {prop: [], 'id': [], 'chunk_id': [], 'chunk': [], 'filename': [],
                         'z_score': [], 'iqr': [], 'outlier': []}})

    for sets in range(len(data_sets)):
        if previous_run is not None:
            if exists(f'tmp/{prop}{exclude_name}.pkl'):
                data = load_var(f'tmp/{prop}{exclude_name}.pkl')
                break
        name = names[sets]
        sets = data_sets[sets]
        print(f'Start data set: {name}')
        pattern = '[0-9.]*.pkl'
        if not sets[-1] == '/':
            pattern = sets.split('/')[-1]
            sets = '/'.join(sets.split('/')[:-1])
        files = list_all_files(data_path + 'cod/data_sets/' + sets, pattern=pattern)
        for file in range(len(files)):
            file = files[file]
            print(f'File: {file}')
            atoms = load_var(file)
            for atom in atoms:
                filename = None
                try:
                    if isinstance(atom, dict):
                        if 'ase' in atom:
                            atom = atom['ase']
                    if isinstance(atom, str):
                        raise Exception('ASE could not read the atomic file')

                    if hasattr(atom, 'filename'):
                        filename = atom.filename
                    elif hasattr(atom, 'filename_full'):
                        filename = atom.filename_full
                    elif 'filename' in atom.keys():
                        filename = atom['filename']
                    if filename is None:
                        raise Exception('Could not find filename')
                    if exclude_condition is not None:
                        if len(set(exclude_condition) & set(atom.get_chemical_symbols())) > 0:
                            continue

                    p = property_fn(atom, **kwargs)
                    data[name][prop].append(p)
                    id = int(filename.split('/')[-1][:7])
                    if not isinstance(p, list):
                        p = [p]
                    for _ in range(len(p)):
                        data[name]['id'].append(id)
                    data[name]['filename'].append(filename)
                    data[name]['chunk_id'].append(int(file.split('/')[-1][:-4].replace('.', '')))
                    data[name]['chunk'].append(file)

                except Exception as e:
                    red_print(e)
                    if str(e) == 'Could not find filename':
                        raise
        # make a flat list if we have a list of lists
        if isinstance(data[name][prop][0], list):
            data[name][prop] = [item for sublist in data[name][prop] for item in sublist]
    xlabel = kwargs.pop('prop', None)

    save_var(data, f'tmp/{prop}{exclude_name}.pkl')
    if return_data:
        return data

    for sets in range(len(data_sets)):
        name = names[sets]

        # Inter-quartile Range (IQR)
        d = data[name][prop]

        if prop == 'atomic_number':
            del data[name]['chunk_id'], data[name]['chunk'], data[name]['filename']
            data[name][prop] = [float(x) for x in data[name][prop]]

        sorted(d)
        q1, q3 = np.percentile(d, [1, threshold_iqr])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        # upper_bound = q3 + (1.5 * iqr)
        upper_bound = q3 + (4 * iqr)
        outliers = np.logical_or(d < lower_bound, d > upper_bound)
        data[name]['iqr'] = outliers
        # compare_plots(d[~outliers], d[outliers])

        # z_score = np.abs(stats.zscore(d))
        # outliers = z_score > threshold_z_score
        data[name]['z_score'] = outliers
        # compare_plots(d[~outliers], d[outliers])

        data[name]['outlier'] = data[name]['iqr']

        data[name] = pandas.DataFrame(data[name])
        if data_sets[sets][-1] == '/':
            file = data_path + 'cod/data_sets/' + '/'.join(data_sets[sets].split('/')[:-1]) + f'/outliers_{prop}.pkl'
            if 'chunk' in data[name].keys():
                del data[name]['chunk']
            save_var(data[name], file)
            save_df(data[name][data[name]['outlier']], file[:-3] + 'txt')
        else:
            red_print('Skipping the outliers.pkl update since it is not going over all the data')
    if prop == 'atomic_number':
        periodic_table = PeriodicTable().table
        x = []
        y = []
        d = []
        for sets in range(len(data_sets)):
            name = names[sets]
            (unique, counts) = np.unique(data[name]['atomic_number'], return_counts=True)
            y += list(counts)
            for i in range((len(unique))):
                print(periodic_table['symbol'][i])
                x.append(periodic_table['symbol'][unique[i] - 1])
                d.append(name)
        table = {'elements': x, 'counts': y, 'data': d}
        table = pandas.DataFrame(table)
        plt.figure(figsize=(20, 20), dpi=600)
        sns.barplot(x='elements', y='counts', data=table, hue='data')
        plt.savefig(f'plots/elements.png', dpi=600)
        plt.savefig(f'plots/elements.svg')
        ind = np.argsort(table['counts'])[-5:]
        plt.figure(figsize=(20, 20), dpi=600)
        sns.barplot(x='elements', y='counts', data=table.drop(table.index[ind]), hue='data')
        plt.savefig(f'plots/elements_no_tops.png', dpi=600)
        plt.savefig(f'plots/elements_no_tops.svg')
        plt.show()
        return

    labels = []
    # plt.close('all')
    for sets in range(len(data_sets)):
        name = names[sets]
        data_val = data[name][prop]

        d = data[name][prop]
        l_i = len(d[data[name]['iqr']])
        l_z = len(d[data[name]['z_score']])
        l_o = len(d[data[name]['outlier']])
        compare_plots(d[data[name]['iqr']], d[data[name]['z_score']],
                      labels=[f'iqr outliers ({threshold_iqr}), Tot. = {l_i}',
                              f'z_score outliers ({threshold_z_score}), Tot. = {l_z}'],
                      bins=[10, 10], title=f'Outliers comparision on \n{name}', xlabel=xlabel,
                      save_name=f'{xlabel}_outliers_comp')

        r = np.logical_and(data_val > x_lim_lo, data_val < x_lim_hi)
        d = data_val[np.logical_and(r, ~data[name]['outlier'])]

        if len(d) == 0:
            raise ValueError('Based on the filters no value can be displayed')

        # x_max = max(max(d) * 1.07, x_max)
        labels.append(f'{name}\nSamples: {len(d)} ({len(d) / len(data_val) * 100:6.2f}%'
                      f'\nMean:{statistics.mean(d):.1f}, Median:{statistics.median(d):.1f}\n'
                      f'Min:{min(d):.1f}, Max:{max(d):.1f})')

        sns.set(style="ticks", rc={"lines.linewidth": 2}, color_codes=True)
        sns.distplot(d, bins=bins[sets],
                     # kde_kws={"lw": 1.5, "label": f'KDE: {name}'},
                     kde=None,
                     norm_hist=normalization,
                     label=labels[sets])
        axes = plt.gca()
        # axes.set_xlim([min(0, min(d)), x_max])
        plt.xlabel(xlabel)
        plt.ylabel('PDF' if normalization else 'Count')
        plt.title(xlabel + f'\nNo outlier included\n {x_lim_lo} < x < {x_lim_hi}')
        plt.legend()
        makedirs('plots/', exist_ok=True)
        plt.savefig(f'plots/{xlabel}_{sets}.png', dpi=600)
        plt.savefig(f'plots/{xlabel}_{sets}.svg')
        plt.show()

    if len(data_sets) == 2:
        compare_plots(data[names[0]][prop][~data[names[0]]['outlier']],
                      data[names[1]][prop][~data[names[1]]['outlier']],
                      labels=labels, xlabel=xlabel,
                      normalization=normalization, bins=bins)


def compare_plots(d1, d2, bins=None, labels=None, xlabel='distribution', normalization=False,
                  x_lim_lo=-1e5, x_lim_hi=1e5, equal_axis=False, title=None, save_name=None):
    if labels is None:
        labels = ['data_1', 'data_2']
    if bins is None:
        bins = [10, 10]
    if title is None:
        title = xlabel
    if save_name is None:
        save_name = xlabel
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    sns.set(style="ticks", rc={"lines.linewidth": 2}, color_codes=True)

    data_val = d1
    d = data_val[np.logical_and(data_val > x_lim_lo, data_val < x_lim_hi)]
    # x_max = max(d)
    sns.distplot(d, bins=bins[0], label=labels[0], ax=ax, color='g', kde=None, norm_hist=normalization)

    # data_val = np.array(data[names[1]]['value'])
    data_val = d2
    d = data_val[np.logical_and(data_val > x_lim_lo, data_val < x_lim_hi)]
    # x_max = max(x_max, max(d))
    sns.distplot(d, bins=bins[1], label=labels[1], ax=ax2, color='r', kde=None, norm_hist=normalization)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.set_xlabel(xlabel)
    # ax.set_xlim(x_max * 1.07)
    lab = 'PDF' if normalization else 'Count'
    ax.set_ylabel(f'{lab} (green)')
    ax2.set_ylabel(f'{lab} (red)')
    if equal_axis:
        ax2.set_ylim(ax.get_ylim())
    plt.title(title + f'\n {x_lim_lo} < x < {x_lim_hi}')
    plt.savefig(f'plots/{save_name}.png', dpi=600)
    plt.savefig(f'plots/{save_name}.svg')
    plt.show()


def cod_id_2_path(cod_id):
    cod_id = str(cod_id)
    return data_path + f'cod/cif/{cod_id[0]}/{cod_id[1:3]}/{cod_id[3:5]}/{cod_id}.cif'


def make_all_the_plots_from_scratch():
    print('Min nearest neighbor')
    distribution(
        data_sets=['all/cif_chunks/', 'anomaly_cspd/cif_chunks/'],
        property_fn=atom2prop,
        prop='min_nearest_neighbor',
        previous_run=True,
        bins=[40, 10],
        x_lim_lo=0,
        x_lim_hi=200,
    )
    print('Density of matrices')
    distribution(
        data_sets=['all_ase/mat_set_128/', 'anomaly_cspd/mat_set_128/'],
        property_fn=atom2prop,
        prop='density-mat',
        previous_run=True,
        bins=[40, 20],
        x_lim_lo=-10,
        x_lim_hi=20,
        equal_axis=True,
    )
    distribution(
        data_sets=['all_ase/mat_set_128/', 'anomaly_cspd/mat_set_128/'],
        property_fn=atom2prop,
        prop='density-mat',
        previous_run=True,
        bins=[40, 10],
        x_lim_lo=10,
        x_lim_hi=200,
        equal_axis=True,
    )


if __name__ == '__main__':
    make_all_the_plots_from_scratch()
    print(f'End of file')
