from utility.util_crystal import *
from cod_tools import *
from keras.models import load_model
from cnn_classifier_3D import CAE
from train_CAE_binary_clf import data_preparation, plot_roc
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, \
    roc_curve, auc
from sklearn.utils import shuffle


def predict_crystal_synthesis(atoms, auto_encoder=None, classifiers=None, verbose=True, pad_len=70, n_bins=128,
                              save_tmp_encoded_atoms=True, redo_calc=False, bulk_calculations=False,
                              parallel_cifs=False, n_jobs=1, convert2df=True, labels=None, skip_svm=True,
                              input_format: str = None,
                              ):
    def single_crystal(atom, encoded_atoms=None):
        path = None
        if (auto_encoder_path is not None) and (isinstance(atom, str)):
            path = '/'.join(atom.split('/')[:-1] + auto_encoder_path.split('/')[-3:-1] + [atom.split('/')[-1] + '.pkl'])
            if exists(path) and (not redo_calc):
                encoded_atoms = load_var(path)

        if isinstance(atom, str):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atom = cif_parser(atom, check_min_dist_cond=True, input_format=input_format,
                                  check_size_of_crystal_cond=True)

        if isinstance(atom, np.ndarray):
            encoded_atoms = atom
            if len(encoded_atoms.shape) == 1:
                encoded_atoms = np.array([encoded_atoms])

        mat3d = None
        if encoded_atoms is None:
            spars_mat = atom2mat(atom, len_limit=pad_len, n_bins=n_bins)
            mat3d = spars_mat_2_3d_mat(spars_mat, 128, ['atomic_number', 'group', 'period'])

        if isinstance(auto_encoder, list):
            encoded_atoms_list = []
            for ae in auto_encoder:
                encoded_atoms_list.append(ae.predict(mat3d))
            encoded_atoms = encoded_atoms_list[0] / len(encoded_atoms_list)
            for d in range(1, len(auto_encoder)):
                encoded_atoms += encoded_atoms_list[d] / len(encoded_atoms_list)
            encoded_atoms = ss.transform(encoded_atoms)
            encoded_atoms = pca.transform(encoded_atoms)
        elif encoded_atoms is None:
            encoded_atoms = auto_encoder.predict(mat3d)
            if save_tmp_encoded_atoms:
                save_var(encoded_atoms, path)

        encoded_atoms = ss.transform(encoded_atoms)
        encoded_atoms = pca.transform(encoded_atoms)

        pred = {}
        for clf_name, classifier in classifiers.items():
            p = classifier.predict_proba(encoded_atoms)[:, 1]
            if len(p) > 1:
                pred.update({clf_name: np.array(np.round(p, decimals=4), dtype='float')})
            else:
                pred.update({clf_name: float(np.round(p, decimals=4))})
        filename = None
        if hasattr(atom, 'filename'):
            filename = atom.filename
        if not bulk_calculations:
            pred.update({'filename': filename})
        return pred

    auto_encoder_path = None
    if auto_encoder is None:
        # auto_encoder = data_path + 'cod/results/CAE/run_043/'
        auto_encoder = 'model/cae/'
    if verbose:
        print(f'CAE: {auto_encoder}')

    if isinstance(auto_encoder, str):
        auto_encoder_path = auto_encoder
        in_size = (n_bins, n_bins, n_bins, 3)
        cae_1 = CAE(input_shape=in_size, pool_size=[4, 4, 2], optimizer='adam', verbose=False)
        auto_encoder = cae_1.generate_encoder(auto_encoder_path + 'weights_1.h5')

    if classifiers is None:
        classifiers = 'model/classifiers/'

    if verbose:
        print(f'clf: {classifiers}')
    if isinstance(classifiers, str):
        classifiers = load_var(classifiers + 'classifiers.pkl')
        if skip_svm:
            classifiers.pop('BaggingClassifier', None)
            classifiers.pop('SVC', None)

    classifiers = classifiers.copy()
    ss = classifiers.pop('StandardScaler')
    pca = classifiers.pop('PCA')
    best_thresholds = {}
    for k in classifiers:
        if 'best_threshold' in k:
            best_thresholds.update({k: classifiers[k]})
    [classifiers.pop(k, None) for k in best_thresholds]

    if not isinstance(atoms, Iterable):
        if isinstance(atoms, str):
            atoms = [atoms]
        else:
            atoms = list(atoms)

    pred_list = []
    if bulk_calculations:
        if not isinstance(atoms, np.ndarray):
            raise ValueError(f'Bulk calculations can be only done when input is'
                             f' array of  atoms passed through descriptor')
        pred_list = single_crystal(atoms)
        pred_list = pd.DataFrame(pred_list)
        return pred_list

    labels_out = []
    for i in range(len(atoms)):
        if verbose:
            print(f'Predicting {i:5}/{len(atoms):5} : ', end='')
        try:
            pred_list.append(single_crystal(atoms[i]))
            if labels is not None:
                labels_out.append(labels[i])
        except Exception as e:
            o = {}
            for c in classifiers:
                o.update({c: None})
            if isinstance(atoms[i], str):
                o.update({'filename': atoms[i]})
            elif hasattr(atoms[i], 'filename'):
                o.update({'filename': atoms[i].filename})
            pred_list.append(o)
            red_print(e)
            e = {}
            for key in classifiers:
                e.update({key: None})
            continue
        if verbose:
            print(pred_list[-1])

    if convert2df:
        pred_list = pd.DataFrame(pred_list)
    if labels is not None:
        pred_list['labels'] = pd.Series(labels_out)
    for c in classifiers:
        t = f'{c}_best_threshold'
        if t in best_thresholds:
            pred_list[t] = best_thresholds[t]
            pred_list[f'{c}_label'] = np.sign(np.sign(pred_list[c] - best_thresholds[t]) + .5)

    # Changing the order of columns
    cols_to_order = ['MLPClassifier', 'MLPClassifier_label',
                     'RandomForestClassifier', 'RandomForestClassifier_label']
    new_columns = cols_to_order + (pred_list.columns.drop(cols_to_order).tolist())
    pred_list = pred_list[new_columns]

    return pred_list


def test_cod_feature_files_results(clf=f'results/Classification/run_043_all_data/classifiers.pkl',
                                   cae_path=f'results/CAE/run_043/',
                                   features='results/features/run_043/',
                                   use_all_data=True
                                   ):
    run_1 = RunSet(ini_from_path=features, new_result_path=True,
                   )
    input_shape = (run_1.n_bins, run_1.n_bins, run_1.n_bins, len(run_1.params['channels']))
    cae = CAE(input_shape=run_1.input_shape, pool_size=[4, 4, 2], optimizer='adam', verbose=False)
    cae = cae.generate_encoder(cae_path + 'weights_1.h5')
    clf = load_var(clf)

    X_train, y_train, X_test, y_test = data_preparation(over_sampling=True, standard_scaler=False, apply_pca=False,
                                                        pca_n_comp=1000, split_data_frac=1., use_all_data=use_all_data,
                                                        pos_frac=1., make_dev_data=False)

    x, y = None, None
    for data_set in ['train', 'test']:
        print('*' * 20, f'\nShowing the results for {data_set} set.')
        x = locals()[f'X_{data_set}']
        y = locals()[f'y_{data_set}']

        predY_prob = predict_crystal_synthesis(x, auto_encoder=cae, classifiers=clf, verbose=True,
                                               pad_len=70, n_bins=128, bulk_calculations=True,
                                               save_tmp_encoded_atoms=False, redo_calc=False)

        predY_prob['Labels'] = pd.Series(y)

        classifiers = [c for c in predY_prob.columns if 'Classifier' in c or 'SVC' in c]

        for c in classifiers:
            print(c)
            print('accuracy = {:.2f}%'.format(100 * accuracy_score(y, np.sign(np.sign(predY_prob[c] - 0.5) - .5))))
            print('auc = {:.2f}%'.format(100 * roc_auc_score(y, predY_prob[c])))

    print('End fn')
    return locals()


def test_cod_cif_results(clf=f'results/Classification/run_043_all_data/classifiers.pkl',
                         cae_path=f'results/CAE/run_043/',
                         use_all_data=True,
                         random_state=1,
                         ):
    run_1 = RunSet(ini_from_path=cae_path, new_result_path=True)
    clf = load_var(clf)

    files_pos = list_all_files(data_path + 'cod/cif/', pattern='**/*.cif', shuffle=True,
                               random_seed=random_state)

    files_neg = list_all_files(data_path + 'cod/anomaly_cspd/cspd_cif_top_108/', pattern='**/*.cif', shuffle=True,
                               random_seed=random_state)
    files_pos = files_pos[:len(files_neg)]
    x = files_pos + files_neg
    y = [1] * len(files_pos) + [-1] * len(files_neg)

    x, y = shuffle(x, y, random_state=random_state)
    n_jobs = get_arg_terminal('cpu', default=4)
    predY_prob = None

    do_predictions = True
    if do_predictions:
        predY_prob = predict_crystal_synthesis(x[:1000], auto_encoder=cae_path, classifiers=clf, verbose=True,
                                               pad_len=70, n_bins=128, bulk_calculations=False,
                                               save_tmp_encoded_atoms=True, redo_calc=False,
                                               parallel_cifs=False, n_jobs=n_jobs, labels=y)
        save_var(predY_prob, run_1.results_path + 'preds.pkl')

    y = np.array(predY_prob['labels'])

    for c in [
        'BaggingClassifier',
        'MLPClassifier',
        'RandomForestClassifier',
    ]:
        print(c)
        print('accuracy = {:.2f}%'.format(100 * accuracy_score(y, np.sign(np.sign(predY_prob - 0.5) - .5))))
        print('auc = {:.2f}%'.format(100 * roc_auc_score(y, predY_prob[c])))

    print('End fn')
    return locals()


if __name__ == '__main__':
    # out = test_cod_feature_files_results(clf=f'results/Classification/run_043/classifiers.pkl',
    #                                      cae_path=f'results/CAE/run_043/',
    #                                      features='results/features/run_043/',
    #                                      use_all_data=False
    #                                      )

    # out = test_cod_cif_results(clf=f'results/Classification/run_043/classifiers.pkl',
    #                            cae_path=f'results/CAE/run_043/',
    #                            use_all_data=True
    #                            )
    # locals().update(out)
    print('The End')
