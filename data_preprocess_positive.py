from crystal_tools import atom2mat, cif_parser
from utility.util_crystal import *
from ase.io import read
from super_cell import run_super_cell_program, prepare_supercell
import multiprocessing
from crystal_tools import spars_mat_2_3d_mat
import urllib.request


def cif2chunk(total_sections=100, query_filter=None, data_set=None,
              output_path='cod/data_sets/all/cif_chunks/', shuffle=True,
              parsers=None, pattern='**/*.cif', random_seed=0, selected_sections=None):
    """
    It get a directory of CIF files and it parses all the CIF to ASE Atoms Object and save them
    into chunks of pickle data sets.

    :param total_sections: Total number of chunks of data
    :param query_filter: To filter some of the CIF files
    :param data_set: The path to the input CIF files directory
    :param output_path: The output path
    :param shuffle: shuffle the data set before splitting to chunks
    :param parsers: ase parser
    :param pattern: pattern by which it recursively goes to children directory looking for CIF files
    :param random_seed:
    :param selected_sections: To create one some of the chunks only. Deterministic random number is necessary in this
    case and can be manipulated by the random_seed variable
    :return: None
    """
    if selected_sections is None:
        selected_sections = range(total_sections)
    if parsers is None:
        parsers = [1e-3, 5e-2, 'ase']
    if data_set is None:
        data_set = ['cod/cif/']
    print(f'Converting cif files to {total_sections} chunks.', flush=True)

    # Filtering the cif files by query
    if query_filter is not None:
        all_files = run_query(query_filter)
        for i in range(len(all_files)):
            d = str(all_files[i][0])
            all_files[i] = data_path + 'cod/cif/' + d[0] + '/' + d[1:3] + '/' + d[3:5] + '/' + d + '.cif'
    else:
        if isinstance(data_set, str):
            data_set = [data_set]
        all_files = []
        for i in data_set:
            all_files = all_files + list_all_files(data_path + i, pattern=pattern, shuffle=shuffle,
                                                   random_seed=random_seed)
            print(i, ' : cumulative number of files = ', len(all_files))

    if shuffle:
        random.Random(0).shuffle(all_files)

    all_files = np.array_split(all_files, total_sections)
    reading = {'readable': [], 'error': []}
    directory = data_path + output_path
    for i in selected_sections:
        print(f'Chunk {i}', flush=True)
        files = all_files[i]
        data = []
        # for j in range(3):
        for j in range(len(files)):
            filename = files[j]
            if not exists(filename):
                raise FileNotFoundError('File were not found')
            di = {}
            for p in parsers:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        atoms_unit = cif_parser(filename, site_tolerance=p)
                    except Exception as e:
                        reading['error'].append(f'{filename} : {str(e)}')
                        di.update({str(p): str(e)})
                        continue
                reading['readable'].append(filename)
                di.update({str(p): atoms_unit})
            data.append(di)
        r1 = len(reading['readable'])
        r2 = len(reading['error'])
        print(f'Chunk {i} completed. Tot. readable = {r1} , Tot. error = {r2}')
        makedirs(directory, exist_ok=True)
        with open(directory + '{:04d}.pkl'.format(i), 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
            pickle_file.close()
            del pickle_file, data
    print('Preparing the results to exit')
    write_text(data_path + output_path + 'readable.txt', reading['readable'])
    write_text(data_path + output_path + 'error.txt', reading['error'])
    summary(directory)


def cif_chunk2mat(total_sections=100, selected_sections=None, pad_len=17.5, n_bins=32, break_into_subsection=1,
                  target_data='cod/data_sets/all/cif_chunks/', output_dataset=None, parser=None,
                  channels=None, make_mat_files=True, n_cpu=1, check_for_outliers=True, npz_sub_sec=None,
                  repairing=False):
    """
    It converts a list of chunks of CIF files to SPARSE 3D cubic images of crystals.
    :param total_sections: Total number of chunks of data
    :param selected_sections: To create one some of the chunks only. Deterministic random number is necessary in this
    case and can be manipulated by the random_seed variable
    :param pad_len: The cube's side length in (A)
    :param n_bins: Number of generated bins in each side of the cube
    :param break_into_subsection: Split each chunk
    :param target_data: The path to the input chunks
    :param output_dataset: The path to the output chunks
    :param parser: ase
    :param channels: None uses the default which is atomic number, group and period for each atom
    :param make_mat_files: Creating the 3D unwrapped images as well (Faster to train but larger to store)
    :param n_cpu: Number of parallel jobs to separately calculate each chunk
    :param check_for_outliers: It removes the outliers
    :param npz_sub_sec: number of splits into sub-chunks when make_mat_files is on
    :return:
    """
    if output_dataset is None:
        output_dataset = '/'.join(target_data.split('/')[:-2]) + f'/mat_set_{n_bins}/'
    if channels is None:
        channels = ['atomic_number', 'group', 'period']
    if isinstance(selected_sections, int):
        selected_sections = [selected_sections]
    output_dir = data_path + output_dataset
    makedirs(output_dir, exist_ok=True)
    if not repairing:
        write_text(output_dir + 'info.txt', dict2str(locals().copy()), makedir=True)

    print('Time: ', str(datetime.now()).split('.')[0])
    print('Converting all cif chunks to matrices.', flush=True)
    print('Total sections = {} \t Sub-sections = {}'.format(total_sections, break_into_subsection))
    print('Selected sections: ', selected_sections, flush=True)
    print('Pad length = {} \t#bins = {}'.format(pad_len, n_bins))

    # outliers = load_var(data_path + 'cod/data_sets/all/cif_chunks/outliers_min_dist_corner.pkl')
    # outliers = np.array(outliers[outliers['outlier']]['id'])
    outliers = [None]

    error = []

    all_files = list_all_files(data_path + target_data, pattern="[0-9]*.pkl")

    if '--range' in sys.argv:
        r1 = int(sys.argv[sys.argv.index('--range') + 1])
        r2 = sys.argv[sys.argv.index('--range') + 2]
        if r2 == ':':
            r2 = len(all_files)
        else:
            r2 = int(r2)
        selected_sections = range(r1, r2)

    if selected_sections is None:
        selected_sections = range(len(all_files))
    df = pandas.DataFrame(columns=['filename', 'n_samples'])
    if not repairing:
        if exists(output_dataset + 'summary.txt'):
            os.remove(output_dataset + 'summary.txt')
            os.remove(output_dataset + 'summary.pkl')
    # for i in range(len(selected_sections)):
    #     calc_chunk(i)

    inp = locals().copy()
    itr = [{'i': i, 'inp': inp} for i in range(len(selected_sections))]
    t1 = datetime.now()

    if n_cpu is None:
        n_cpu = 1
    if '--cpu' in sys.argv:
        n_cpu = int(sys.argv[sys.argv.index('--cpu') + 1])
        if n_cpu == -1:
            n_cpu = multiprocessing.cpu_count()
    print(f'Using {n_cpu} processors.')

    run_in_parallel(cif_chunk2mat_helper, itr, n_jobs=n_cpu)

    # pool = multiprocessing.Pool(processes=n_cpu)
    # pool.map(cif_chunk2mat_helper, itr)
    # pool.close()
    # pool.join()

    dt = datetime.now() - t1
    print('Mat chunks were made. Total time: {},'.format(str(dt).split('.')[0]), flush=True)

    # write_text(output_dir + 'error.txt', error)
    if not repairing:
        summary(output_dir)


def cif_chunk2mat_helper(itr=None):
    """
    The helper function to cif_chunk2mat for parallelization.
    :param itr:
    :return:
    """
    i_sel = None
    try:
        inp = itr['inp']
        all_files = inp['all_files']
        selected_sections = inp['selected_sections']
        break_into_subsection = inp['break_into_subsection']
        pad_len = inp['pad_len']
        n_bins = inp['n_bins']
        output_dir = inp['output_dir']
        parser = inp['parser']
        channels = inp['channels']
        make_mat_files = inp['make_mat_files']
        outliers = inp['outliers']
        check_for_outliers = inp['check_for_outliers']
        npz_sub_sec = inp['npz_sub_sec']

        if npz_sub_sec is not None:
            if not isinstance(npz_sub_sec, list):
                npz_sub_sec = [npz_sub_sec]

        # for k, v in itr['inp'].items():
        #     locals()[k] = v
        error = []
        i_sel = itr['i']
        print('Opening file #', i_sel, ' :', all_files[selected_sections[i_sel]], flush=True)
        loaded_files = np.array_split(load_var(all_files[selected_sections[i_sel]]), break_into_subsection)

        for sub in range(len(loaded_files)):
            gc.collect()
            files = loaded_files[sub]
            data = []
            t1 = datetime.now()
            for j in range(len(files)):
                if not make_mat_files:
                    if exists(output_dir + '{:03d}.{:03d}.pkl'.format(selected_sections[i_sel], sub)):
                        print('Lading pre-made data: {:03d}.{:03d}.pkl'.format(selected_sections[i_sel], sub))
                        data = load_var(output_dir + '{:03d}.{:03d}.pkl'.format(selected_sections[i_sel], sub))
                        break
                    else:
                        make_mat_files = True
                        print('Although the request, the mat file dose not exist and it should be made')
                atoms = files[j]
                if isinstance(atoms, dict):
                    if parser is None:
                        raise ValueError('You have to specify the atomic parser')
                    atoms = atoms[parser]
                filename = f'e/Chunk: {all_files[selected_sections[i_sel]]}, Object#: {j}'
                try:
                    if isinstance(atoms, str):
                        raise Exception('ASE could not read the atomic file')
                    if not hasattr(atoms, 'filename'):
                        atoms.filename = atoms.filename_full
                        filename = atoms.filename
                    if check_for_outliers:
                        if int(filename.split('/')[-1].replace('_', '.').split('.')[0]) in outliers:
                            raise Exception('This is an outlier.')
                    d = atom2mat(filename='', len_limit=pad_len, n_bins=n_bins, atoms_unit=atoms)
                except Exception as e:
                    error.append('{} : {}'.format(filename.split('/')[-1], str(e)))
                    red_print(e)
                    continue
                data.append(d)
            if make_mat_files:
                print('Saving sub-section', output_dir + '{:03d}.{:03d}.pkl'.format(selected_sections[i_sel], sub))
                save_var(data, output_dir + '{:03d}.{:03d}.pkl'.format(selected_sections[i_sel], sub), )
            print(f'Converting {selected_sections[i_sel]:03d}.{sub:03d}.pkl to 3D matrices.'
                  f'\nTot. samples={len(data)}, Dividing to {len(data) // 50}'
                  f' sub. sec. with ~50 samples in each.', flush=True)
            data = np.array_split(data, max(1, len(data) // 50))
            makedirs(output_dir + f'{selected_sections[i_sel]:03d}/', exist_ok=True)
            if npz_sub_sec is None:
                for i in range(len(data)):
                    mat3 = spars_mat_2_3d_mat(list(data[i]), n_bins, channels)
                    print(f'Saving sub sec. {selected_sections[i_sel]:03d}/{i:03d}.npz')
                    np.savez_compressed(output_dir + f'{selected_sections[i_sel]:03d}/{i:03d}.npz', mat3)
            else:
                print('Going over selected npz subsections: ')
                print(npz_sub_sec, flush=True)
                for i in npz_sub_sec:
                    mat3 = spars_mat_2_3d_mat(list(data[i]), n_bins, channels)
                    print(f'Saving sub sec. {selected_sections[i_sel]:03d}/{i:03d}.npz')
                    np.savez_compressed(output_dir + f'{selected_sections[i_sel]:03d}/{i:03d}.npz', mat3)

            dt = datetime.now() - t1
            print('Section calc. time: {},'.format(str(dt).split('.')[0]))
            write_text(output_dir + 'error.{:03d}.{:03d}.txt'.format(selected_sections[i_sel], sub), error)
        progress = len(list_all_files(output_dir, pattern="[0-9]*.pkl"))
        print(f'Pooling progress: {progress}/{len(all_files)}', flush=True)
    except Exception as e:
        red_print(f'Process {i_sel} crashed with error: {str(e)}')
        print_exception()
        save_var(locals(), f'tmp/multi_proc_{i_sel}.pkl')


def chunk_filter(target_data='cod/data_sets/all/cif_chunks/', output_dataset=None):
    print('Time: ', datetime.now())
    print('Converting all cif chunks to matrices.', flush=True)
    all_files = list_all_files(data_path + target_data, pattern="[0-9]*.pkl")
    selected_sections = range(len(all_files))
    df = pandas.DataFrame(columns=['filename', 'n_samples'])
    error = []
    for i in range(len(selected_sections)):
        print('Opening file #', i, ' :', all_files[selected_sections[i]])
        files = load_var(all_files[selected_sections[i]])
        data = []
        for j in range(len(files)):
            d = files[j]
            if len(re.findall('super', d.filename.split('/')[-1])) > 0:
                d.super_cell = True
            else:
                d.super_cell = False
            if d.super_cell:
                error.append(d.filename)
                continue
            data.append(d)
        df = df.append({'filename': all_files[selected_sections[i]], 'n_samples': len(data)}, ignore_index=True)
        print('Tot. passed: {} , Tot. eliminations {}'.format(sum(df['n_samples']), len(error)))
        if output_dataset is not None:
            output_dir = data_path + output_dataset
            makedirs(output_dir, exist_ok=True)
            print('Saving sub-section', output_dir + '{:04d}.pkl'.format(selected_sections[i]))
            save_var(data, output_dir + '{:03d}.pkl'.format(selected_sections[i]))
    df['CUM_SUM'] = df['n_samples'].cumsum()
    print_df(df)
    if output_dataset is not None:
        output_dir = data_path + output_dataset
        summary(output_dir)


def filter_cif_with_partial_occupancy(verbose=True):
    """
    It goes over the COD crystals and saves a list of structures with a partial occupancy to supercell_list.sh
    :param verbose:
    :return: None
    """
    def check_occ(a):
        occ = False
        if 'occupancy' not in a.info:
            return occ
        for k in a.info['occupancy'].keys():
            o = a.info['occupancy'][k][list(a.info['occupancy'][k].keys())[0]]
            if not o == 1:
                occ = True
                break
        return occ

    if verbose:
        print('Checking the occupancies', flush=True)
    files = list_all_files(data_path + 'cod/cif/', pattern='**/*.cif')
    bash = []
    for f in range(len(files)):
        if f in np.floor(np.linspace(0, len(files), 1000)):
            print(f'{f} out of {len(files)}')
        file = files[f]
        # noinspection PyBroadException
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms = read(file)
        except Exception as e:
            print(f'ASE error in reading. {e}')
            continue
        occupancy = check_occ(atoms)
        output_path = '/cif_supercell/'.join(re.split("/cif/", file))
        if occupancy:
            output_path = output_path[:-4] + '-supercell.cif'
            super_cell_executable = expanduser('~/Apps/supercell.linux/supercell')
            command = f'{super_cell_executable} -i {file} -o {output_path}'
            bash.append(command)
        else:
            makedirs(output_path[:-11], exist_ok=True)
            shutil.copyfile(file, output_path)
    write_text('supercell_list.sh', '\n'.join(bash))


def load_data(path=None, verbose=True, test_size=0.3, limited_dataset=False, pos_frac=1.,
              random_state=1):
    """
    Loads the crystal space representation of whole database and prepares test and train sets, but it doesn't do
    any processes like SS, PCA or Over-Sampling

    :param path: Path to the data files
    :param verbose: True
    :param test_size: The ratio of test set
    :param limited_dataset: Use it for testing purposes only (It doesn't prepare all the structures)
    :param pos_frac: Only loads a portion of positive set
    :param random_state: The random machine state
    :return: X_train, y_train, X_test, y_test
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    if path is None:
        path = data_path + 'cod/results/features/run_043/'

    if verbose:
        print(f'Loading features from: {path}', flush=True)

    # global X_train, y_train, X_test, y_test

    pos_train = pos_test = neg_x = None

    if limited_dataset:
        red_print('Loading the limited features')
        pos_train = load_var(path + 'pos_train_features.pkl.npz', uncompress=True)
        pos_test = load_var(path + 'pos_test_features.pkl.npz', uncompress=True)
        neg_x = load_var(path + 'neg_features.pkl.npz', uncompress=True)

    if not limited_dataset:
        print('Loading all the features.\nNegative and training', flush=True)
        neg_x = load_var(path + 'features/batch_neg.pkl.npz', uncompress=True)
        print(f'Tot Negative = {len(neg_x):,}')
        pos_files = list_all_files(path + 'features/', pattern='batch_pos**')
        pos = []
        pos_train = []
        if exists(path.replace('features', 'CAE')):
            pos_train = load_var(path.replace('features', 'CAE') + 'pos_train_features.pkl.npz',
                                 uncompress=True)
            pos.append(
                load_var(path.replace('features', 'CAE') + 'pos_test_features.pkl.npz', uncompress=True))
        for f in pos_files:
            print(f'Loading: {f}', end='')
            pos.append(load_var(f, uncompress=True))
            print(f' Samples = {len(pos[-1]):,}', flush=True)
        pos = np.vstack(pos)

        if pos_frac < 1:
            print(f'Using {pos_frac} of total POSITIVE data')
            pos, _ = train_test_split(pos, test_size=1 - pos_frac, random_state=random_state)
        pos, pos_test = train_test_split(pos,
                                         test_size=1 - ((1 - test_size) * len(pos) - test_size * len(
                                             pos_train)) / (len(pos)),
                                         random_state=random_state)
        pos = np.vstack([pos, pos_train])
        pos_train = pos
        print(f'Tot Positive = {len(pos_train) + len(pos_test):,}', flush=True)
        # Average features by combining features
    average_cae = False
    if average_cae:
        p1 = load_var('results/CAE/run_041/pos_test_corr.pkl.npz', uncompress=True)
        p2 = load_var('results/CAE/run_043/pos_test_corr.pkl.npz', uncompress=True)
        p3 = load_var('results/CAE/run_045/pos_test_corr.pkl.npz', uncompress=True)

        p_train = (p1 + p2 + p3) / 3
        pos_train, pos_test = train_test_split(p_train, test_size=0.3, random_state=random_state)
        neg_x = neg_x[:1551]

    average_weights = False
    if average_weights:
        red_print('Average weights')
        p_train = load_var('results/CAE/run_041/pos_test_corr.pkl.npz', uncompress=True)
        pos_train, pos_test = train_test_split(p_train, test_size=0.3, random_state=random_state)
        neg_x = load_var('results/CAE/run_041_045/neg_features.pkl')
    # N = 3
    N = 1

    if N > 1:
        red_print(f'Running on a portion of data - Divided by {N}')
        pos_train = pos_train[:len(pos_train) // N]
        pos_test = pos_test[:len(pos_test) // N]
        neg_x = neg_x[:len(neg_x) // N]

    if verbose:
        print('Splitting to train and test', flush=True)
    neg_train, neg_test = train_test_split(neg_x, test_size=test_size, random_state=random_state)

    X_train = np.concatenate((pos_train, neg_train), axis=0)
    y_train = np.concatenate(([1] * len(pos_train), [-1] * len(neg_train)), axis=0)
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

    X_test = np.concatenate((pos_test, neg_test), axis=0)
    y_test = np.concatenate(([1] * len(pos_test), [-1] * len(neg_test)), axis=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

    return X_train, y_train, X_test, y_test


def data_preparation(over_sampling=True, standard_scaler=True, apply_pca=True, pca_n_comp=1000, split_data_frac=1.,
                     pos_frac=1., make_dev_data=False, use_all_data=True, random_state=1,
                     path=None, return_pca_ss=False):
    """
    Loads the train and test sets and it applies Standard Scalar, PCA and OverSampling.

    :param over_sampling: Does Over Sampling if True
    :param standard_scaler: Applies Standard Scaler if True
    :param apply_pca: Applies PCA if True
    :param pca_n_comp: Number of components to keep applying PCA
    :param split_data_frac: Use a fraction of data
    :param pos_frac: Use a fraction of positive set
    :param make_dev_data: Create development set in addition to test and train
    :param use_all_data: Prepares all the data
    :param random_state: Machine's random state
    :param path: Location of data sets
    :param return_pca_ss: Request the PCA and SS methods to be returned
    :return: X_train, y_train, X_test, y_test
    or
    X_train, y_train, X_test, y_test, X_dev, y_dev
    or
    Dictionary
    """
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import RandomOverSampler

    # global X_train, y_train, X_test, y_test, run_1, X_dev, y_dev
    # global meta_data, ss, pca
    X_dev = y_dev = None
    X_train, y_train, X_test, y_test = load_data(path=path, limited_dataset=not use_all_data, pos_frac=pos_frac)

    meta_data = ["Stats before over sampling"]
    print(meta_data[-1])
    meta_data += ['Total train Samples = {:,} \t({:,} ({:.1f}%) '
                  'Positive)'.format(len(y_train), np.count_nonzero(y_train == 1),
                                     np.count_nonzero(y_train == 1) / len(y_train) * 100)]
    print(meta_data[-1])
    meta_data += ['Total test Samples  = {:,} \t({:,} ({:.1f}%) '
                  'Positive)'.format(len(y_test), np.count_nonzero(y_test == 1),
                                     np.count_nonzero(y_test == 1) / len(y_test) * 100)]
    print(meta_data[-1])

    if over_sampling:
        print('Over Sampling ... ', flush=True)
        ros = RandomOverSampler(random_state=0)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_test, y_test = ros.fit_resample(X_test, y_test)

        meta_data += ['\n']
        meta_data += ["Stats after over sampling"]
        print(meta_data[-1])
        meta_data += ['Total train Samples = {:,} \t({:,} ({:.1f}%) '
                      'Positive)'.format(len(y_train), np.count_nonzero(y_train == 1),
                                         np.count_nonzero(y_train == 1) / len(y_train) * 100)]
        print(meta_data[-1])
        meta_data += ['Total test Samples  = {:,} \t({:,} ({:.1f}%) '
                      'Positive)'.format(len(y_test), np.count_nonzero(y_test == 1),
                                         np.count_nonzero(y_test == 1) / len(y_test) * 100)]
        print(meta_data[-1])

    if split_data_frac < 1:
        print(f'Using {split_data_frac} of total data')
        X_train, _, y_train, _ = train_test_split(X_train, y_train,
                                                  test_size=1 - split_data_frac,
                                                  random_state=random_state)
        X_test, _, y_test, _ = train_test_split(X_test, y_test,
                                                test_size=1 - split_data_frac,
                                                random_state=random_state)
        meta_data += ['\n']
        meta_data += [f"Stats after picking {split_data_frac} of data"]
        print(meta_data[-1])
        meta_data += ['Total train Samples = {:,} \t'
                      '({:.1f}% Positive)'.format(len(y_train),
                                                  np.count_nonzero(y_train == 1) / len(y_train) * 100)]
        print(meta_data[-1])
        meta_data += ['Total test Samples  = {:,} \t'
                      '({:.1f}% Positive)'.format(len(y_test), np.count_nonzero(y_test == 1) / len(y_test) * 100)]
        print(meta_data[-1])

    if make_dev_data:
        meta_data += [f'0.3 of train set is using as the development set']
        print(meta_data[-1])
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,
                                                          test_size=0.3,
                                                          random_state=random_state)

    if standard_scaler:
        print('Fit and transform by standard scalar', flush=True)
        ss = StandardScaler()
        ss.fit(X_train)
        X_train = ss.transform(X_train)
        X_test = ss.transform(X_test)
        if make_dev_data:
            X_dev = ss.transform(X_dev)

    pca = ss = None
    if apply_pca:
        print(f'Fit and transform by PCA (n_comp = {pca_n_comp:,})', flush=True)

        meta_data += [f'Effective PCA components: {pca_n_comp:,}']
        pca = PCA(n_components=pca_n_comp, whiten=True, random_state=0)

        pca = pca.fit(X_train)
        print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        if make_dev_data:
            X_dev = pca.transform(X_dev)

    if return_pca_ss:
        return {'data': (X_train, y_train, X_test, y_test),
                'ss': ss, 'pca': pca, }
    if make_dev_data:
        return X_train, y_train, X_test, y_test, X_dev, y_dev
    return X_train, y_train, X_test, y_test


def prepare_data_from_scratch():
    """
    It prepares all the positive data from scratch. If one of the processors is being killed the command line prompts
    that are being run by run_bash_commands can be directly use in the command line in the the same directory.
    :return:
    """
    if not exists(data_path + 'cod-cifs-mysql.tgz'):
        print(f'Downloading the database to: {data_path}cod-cifs-mysql.tgz', flush=True)
        url = 'http://www.crystallography.net/archives/cod-cifs-mysql.tgz'
        urllib.request.urlretrieve(url, data_path + 'cod-cifs-mysql.tgz')
        print('Downloading the dataset is done!', flush=True)

    if not exists(data_path + 'cod'):
        print(f'Extracting the database: cod-cifs-mysql.tgz', flush=True)
        run_bash_commands(f'tar -zxvf cod-cifs-mysql.tgz', path=data_path, verbose=True, wait=True)
        print('Extracting completed', flush=True)
        input('Is the extracting process completed?')

    prepare_supercell()
    filter_cif_with_partial_occupancy()
    run_super_cell_program()

    cif2chunk(data_set=['cod/cif_no_occupancy/', 'cod/cif_supercell/'],
              total_sections=100, shuffle=True)

    for i in [4]:
        cif_chunk2mat(target_data='cod/data_sets/all/cif_chunks/',
                      # output_dataset=
                      n_bins=32 * i, pad_len=17.5 * i, parser='ase'
                      )

        cif_chunk2mat(target_data='cod/data_sets/anomaly_cspd/cif_chunks/',
                      n_bins=32 * i, pad_len=17.5 * i, parser='ase'
                      )


if __name__ == "__main__":
    time_start = datetime.now()
    gc.enable()
    import psutil
    print('Total available memory: {}'.format(human_readable(psutil.virtual_memory().available)), flush=True)

    prepare_data_from_scratch()

    time_end = datetime.now()
    print('End\t', str(time_end - time_start).split('.')[0])
