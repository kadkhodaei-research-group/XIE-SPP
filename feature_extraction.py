from utility.util_general import *
from keras.models import load_model
from cnn_classifier_3D import *
from samplegenerator import *


def feature_extraction(samples_path=None, generator=None, run: RunSet = None, frac=1., auto_encoder=None,
                       verbose=1, pre_made_mat=False, steps=None, workers=4, test=False):
    if auto_encoder is None:
        auto_encoder = load_model(run.params['model_path'] + 'model.h5')
    for i in range(len(auto_encoder.layers)):
        if ('encode' in auto_encoder.layers[-1].name) or ('pooling' in auto_encoder.layers[-1].name):
            break
        auto_encoder.pop()
    auto_encoder.add(layers.Flatten())
    if generator is None:
        if samples_path is None:
            samples_path = run.params.pop('neg_x', None)
            if samples_path is None:
                raise ValueError('sample path can not be empty')
        if not samples_path[0] == '/':
            samples_path = data_path + 'cod/data_sets/' + samples_path
        print('Making the generator')
        tot_pos = summary(samples_path)
        if pre_made_mat:
            files = list_all_files(samples_path, pattern='[0-9]*/[0-9.]*.npz')
            tot_pos = pandas.DataFrame({'filename': files,
                                        'n_samples': [tot_pos['CUM_SUM'].iloc[-1] // len(files)] * len(
                                            files)})
        tot_pos = tot_pos.sample(frac=frac, random_state=0)
        tot_pos.reset_index(inplace=True, drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tot_pos.loc[:, 'CUM_SUM'] = tot_pos['n_samples'].cumsum()
        print_df(tot_pos)
        generator = SampleGenerator(filename=tot_pos['filename'],
                                    pad_len=run.pad_len,
                                    n_bin=run.n_bins,
                                    verbose=0,
                                    sub_sec=run.batch_sub_sec,
                                    channels=run.params['channels']
                                    )

    print(f'Extracting features for {generator.name}', flush=True)
    # generator.__getitem__(717)

    if steps is None:
        steps = len(generator)
    if test:
        red_print('This is a test run')
        steps = 2

    X = auto_encoder.predict_generator(generator,
                                       steps=steps,
                                       use_multiprocessing=True,
                                       workers=workers,
                                       verbose=verbose,
                                       )
    return X


def whole_dataset_feature_extraction(run: RunSet):
    """
    Given a trained CAE model this function extract all the corresponding features.
    :param run:
    :return:
    """
    tot_samples_df = summary(run.chunks_path)
    if run.params.get('pre_made_mat') is True:
        files = list_all_files(run.chunks_path, pattern='[0-9]*/[0-9.]*.npz')
        random.Random(run.params['random_seed']).shuffle(files)
        tot_samples_df = pandas.DataFrame({'filename': files,
                                           'n_samples': [tot_samples_df['CUM_SUM'].iloc[-1] // len(files)] * len(
                                               files)})
    tot_samples_df = np.array_split(tot_samples_df, int(1 / run.params['samples_fraction']))

    del tot_samples_df[run.params['batch']]
    cae = CAE(input_shape=run.input_shape, pool_size=[4, 4, 2], optimizer='adam')
    auto_encoder = cae.generate()
    auto_encoder.load_weights(run.last_run + 'weights_1.h5')

    skip_negative = True
    if not skip_negative:
        print('Extracting negative features...')
        neg_x = feature_extraction(run.params['neg_x'], run=run, auto_encoder=auto_encoder, frac=1)
        save_var(neg_x, run.results_path + f'features/batch_neg.pkl', compress_np=True)
        del neg_x

    batch = get_arg_terminal('batch')
    tot_batch = get_arg_terminal('tot_batch', default=6)
    if tot_batch == -1:
        tot_batch = len(tot_samples_df)
    if batch is not None:
        print(f'Dividing to total batches into {tot_batch} and selecting #{batch}')
        print(np.array_split(range(len(tot_samples_df)), tot_batch))
        print('selected: ', np.array_split(range(len(tot_samples_df)), tot_batch)[batch])

    for i in range(len(tot_samples_df)):
        print(f'Extracting positive features batch {i}/{len(tot_samples_df)}', flush=True)
        if batch is not None:
            if i not in np.array_split(range(len(tot_samples_df)), tot_batch)[batch]:
                red_print('Skipping the batch')
                continue

        generator = SampleGenerator(
            filename=tot_samples_df[i]['filename'].reset_index(drop=True),
            pad_len=run.pad_len,
            n_bin=run.n_bins,
            verbose=0,
            # verbose=2,  # ???????????
            sub_sec=run.batch_sub_sec,
            # test_run=True,
            channels=run.params['channels'],
            name='train_generator'
        )

        try:
            x_features = feature_extraction(generator=generator, run=run, auto_encoder=auto_encoder, workers=-1)
            print('Saving the results to', run.results_path + f'features/batch_pos_{i:02d}.pkl')
            save_var(x_features, run.results_path + f'features/batch_pos_{i:02d}.pkl', compress_np=True)
            del x_features
        except Exception as e:
            red_print(f'Error calculating batch {i}/{len(tot_samples_df)}')
            red_print(e)
            print('Skipping to the next batch...')
            continue
    print('End Fn')


def combine_data_sets():
    run_1 = RunSet(ini_from_path='results/CAE/run_041/', params={'skip_error': True})
    run_2 = RunSet(ini_from_path='results/CAE/run_045/', params={'skip_error': True, 'run_id': run_1.run_id})

    print('Finding the average features of run 41 and 43 by avg weight')

    cae = CAE(input_shape=run_1.input_shape, pool_size=[4, 4, 2], optimizer='adam')
    auto_encoder1 = cae.generate()
    auto_encoder2 = cae.generate()
    auto_encoder = cae.generate()
    auto_encoder1.load_weights(run_1.last_run + 'weights_1.h5')
    auto_encoder2.load_weights(run_1.last_run + 'weights_1.h5')
    models = [auto_encoder1, auto_encoder2]

    weights = [model.get_weights() for model in models]

    new_weights = list()

    for weights_list_tuple in zip(*weights):
        new_weights.append(
            [np.array(weights_).mean(axis=0) \
             for weights_ in zip(*weights_list_tuple)])
    auto_encoder.set_weights(new_weights)

    files = list_all_files(run_1.chunks_path, pattern='[0-9]*/[0-9.]*.npz')
    random.Random(20).shuffle(files)
    files = files[:92]
    gen = SampleGenerator(filename=files,
                          pad_len=run_1.pad_len,
                          n_bin=run_1.n_bins,
                          verbose=0,
                          sub_sec=run_1.batch_sub_sec,
                          channels=run_1.params['channels'],
                          name='generator'
                          )
    pos_test_1 = feature_extraction(generator=gen, run=run_1, auto_encoder=auto_encoder)
    save_var(pos_test_1, run_1.results_path + 'pos_test_corr.pkl', compress_np=True)

    neg_x = feature_extraction(run_1.params['neg_x'], run=run_1, auto_encoder=auto_encoder, frac=0.8,
                               pre_made_mat=True)
    save_var(neg_x, run_1.results_path + 'neg_features.pkl')

    auto_encoder.save(run_1.results_path + f'model_{run_1.num_epochs}.h5')
    auto_encoder.save_weights(run_1.results_path + f'weights_{run_1.num_epochs}.h5')
    save_var(run_1, run_1.results_path + 'run.pkl')

    print("Correlation features extracted")
    print('End Fn')


if __name__ == '__main__':
    extract_whole_data_set = False
    if extract_whole_data_set:
        run_1 = RunSet(ini_from_path=get_arg_terminal('run_from', default='results/CAE/run_043/'))
        whole_dataset_feature_extraction(run_1)
        # combine_data_sets()
        run_1.end()
    run_m = RunSet(params={'random_seed': 1})

    extract_common_batches = False
    if extract_common_batches:
        common_batches = [9, 8]
        cae_number = get_arg_terminal('cae_n', default=45)
        path = f'{data_path}cod/results/CAE/run_{cae_number:03d}/'
        print(f'Extracting common features for: {cae_number}', flush=True)
        run_old = load_var(f'{path}run.pkl')
        files1 = list_all_files(data_path + run_old.chunks_path.split('/Data/')[1], pattern='[0-9]*/[0-9.]*.npz')
        random.Random(run_old.params['random_seed']).shuffle(files1)
        tot_samples_df1 = summary(data_path + run_old.chunks_path.split('/Data/')[1])
        tot_samples_df1 = pandas.DataFrame({'filename': files1,
                                           'n_samples': [tot_samples_df1['CUM_SUM'].iloc[-1] // len(files1)] * len(
                                               files1)})
        tot_samples_df1 = np.array_split(tot_samples_df1, int(1 / run_old.params['samples_fraction']))
        common_batch = tot_samples_df1[common_batches[0]]
        common_batch.reset_index(inplace=True, drop=True)
        common_generator = SampleGenerator(filename=common_batch['filename'],
                                           pad_len=run_old.pad_len,
                                           n_bin=run_old.n_bins,
                                           verbose=0,
                                           # verbose=2,  # ???????????
                                           sub_sec=run_old.batch_sub_sec,
                                           # test_run=True,
                                           channels=run_old.params['channels'],
                                           name='common_generator'
                                           )
        run_m.test_generator = common_generator
        weights_file = list_all_files(path, pattern='weights_epoch*')[-1]
        auto_encoder_1 = CAE(input_shape=run_old.input_shape, pool_size=[4, 4, 2])
        auto_encoder_1 = auto_encoder_1.generate()
        auto_encoder_1.load_weights(weights_file)

        X_common = feature_extraction(generator=common_generator, auto_encoder=auto_encoder_1, test=False)
        print('Saving the results to:')
        print(path + f'features/batch_pos_{common_batches[0]:02d}.pkl', flush=True)
        save_var(X_common, path + f'features/batch_pos_{common_batches[0]:02d}.pkl', compress_np=True)

        run_m.end()
