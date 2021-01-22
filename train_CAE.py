from cnn_classifier_3D import *
from samplegenerator import SampleGenerator
from sklearn.model_selection import train_test_split
from utility.util_tf import *
from utility.util_crystal import *
from keras.callbacks import ModelCheckpoint  # , History
from feature_extraction import feature_extraction, whole_dataset_feature_extraction


def train_cae(batch=0):
    comments = '''3 channels: Atomic number#, Group, Period
    model 5 
    '''
    n = 4
    run = RunSet(params={'pad_len': 17.5 * n,
                         'n_bins': 32 * n,
                         'data_set': f'all_ase/mat_set_{32 * n}/',
                         'neg_x': f'anomaly_cspd/mat_set_{32 * n}/',
                         'pre_made_mat': True,
                         'num_epochs': 10,
                         'batch_sub_sec': 25,  # To resolve memory issue go for higher values
                         'samples_fraction': .05 / 25.,
                         'random_seed': 1,
                         'loss': 'binary_crossentropy',
                         'channels': ['atomic_number', 'group', 'period'],
                         'conv_type': 'cae',
                         'comments': comments,
                         },
                 )

    print(datetime.now)
    print(run)

    if '--random' in sys.argv:
        rnd = int(sys.argv[sys.argv.index('--random') + 1])
        if rnd == -1:
            print('Randomly picking the random seed')
            rnd = random.randint(0, int(1e3))
        run.params['random_seed'] = rnd

    if 'random_seed' in run.params.keys():
        rnd = run.params['random_seed']
        random_seed(rnd)
        print(f'Random seed set to: {rnd}')

    if run.train_generator is None:
        input_shape = (run.n_bins, run.n_bins, run.n_bins, len(run.params['channels']))
        run.input_shape = input_shape

        tot_samples_df = summary(run.chunks_path)
        if run.params.get('pre_made_mat') is True:
            files = None
            if run.n_bins == 128:
                files = list_all_files(run.chunks_path, pattern='[0-9]*/[0-9.]*.npz')
            elif run.n_bins < 128:
                files = list_all_files(run.chunks_path, pattern='[0-9.]*.pkl')

            random.Random(run.params['random_seed']).shuffle(files)
            tot_samples_df = pandas.DataFrame({'filename': files,
                                               'n_samples': [tot_samples_df['CUM_SUM'].iloc[-1] // len(files)] * len(
                                                   files)})

        tot_samples_df = np.array_split(tot_samples_df, int(1 / run.params['samples_fraction']))
        tot_neg_samples_df = summary(data_path + 'cod/data_sets/anomaly_cspd/mat_set_128/')
        if run.params['conv_type'].lower() == 'cae':
            files = list_all_files(data_path + 'cod/data_sets/' + run.params['neg_x'], pattern='[0-9]*/[0-9.]*.npz')
            random.Random(run.params['random_seed']).shuffle(files)
            tot_neg_samples_df = pandas.DataFrame({'filename': files,
                                                   'n_samples': [58] * len(
                                                       files)})
            tot_neg_samples_df = tot_neg_samples_df[:10]
            # tot_samples_df

        if '--batch' in sys.argv:
            batch = int(sys.argv[sys.argv.index('--batch') + 1])
            if batch == -1:
                batch = np.random.randint(0, len(tot_samples_df))
                print('Randomly picking a batch of files for training')
        run.params['batch'] = batch
        print(f'Batch files #{batch} was picked for training')

        tot_samples_df = tot_samples_df[run.params['batch']]
        if run.params['conv_type'].lower() == 'cae':
            tot_neg_samples_df['label'] = 0
            tot_samples_df['label'] = 1
            tot_samples_df = pd.concat([tot_samples_df, tot_neg_samples_df])

        num_total_samples = sum(tot_samples_df['n_samples'])
        train_x, test_x = train_test_split(tot_samples_df, test_size=0.3, random_state=run.params['random_seed'])
        if 'test' in run.params.keys():
            if run.params['test'] is True:
                train_x = train_x[:1]
                test_x = test_x[:1]
        train_x.reset_index(inplace=True, drop=True)
        test_x.reset_index(inplace=True, drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x.loc[:, 'CUM_SUM'] = train_x['n_samples'].cumsum()
            train_x.loc[:, 'CUM_SUM'] = train_x['n_samples'].cumsum()
            tot_samples_df.loc[:, 'CUM_SUM'] = tot_samples_df['n_samples'].cumsum()
        run.train_x = train_x
        run.test_x = test_x
        print('Total training = {}\t Total test = '
              '{}'.format(sum(run.train_x['n_samples']), sum(run.test_x['n_samples'])))
        print('Tot # of samples = {}, Tot # batches (files) '
              '= {}, Each batch divides to {} sections, # epochs = {}'.format(
                num_total_samples, len(tot_samples_df), run.batch_sub_sec, run.num_epochs))
        print(f'Samples per sub-batch ~ {num_total_samples / len(tot_samples_df) / run.batch_sub_sec:.1f}')
        print('#Bins = {} \t Pad len = {} \t '.format(run.n_bins, run.pad_len))

        # Generator
        print('Making the generator')
        train_generator = SampleGenerator(filename=run.train_x['filename'],
                                          labels=run.train_x['label']
                                          if run.params['conv_type'].lower() == 'cae' else None,
                                          pad_len=run.pad_len,
                                          n_bin=run.n_bins,
                                          verbose=0,
                                          sub_sec=run.batch_sub_sec,
                                          channels=run.params['channels'],
                                          name='train_generator'
                                          )

        test_generator = SampleGenerator(filename=run.test_x['filename'],
                                         labels=run.test_x['label']
                                         if run.params['conv_type'].lower() == 'cae' else None,
                                         pad_len=run.pad_len,
                                         n_bin=run.n_bins,
                                         verbose=0,
                                         sub_sec=run.batch_sub_sec,
                                         channels=run.params['channels'],
                                         name='test_generator'
                                         )

        run.test_generator = test_generator
        run.train_generator = train_generator
        print('Generators were created')
    if not hasattr(run, 'input_shape'):
        input_shape = (run.n_bins, run.n_bins, run.n_bins, len(run.params['channels']))
        run.input_shape = input_shape
    auto_encoder = None
    initial_epoch = 0
    if not hasattr(run, 'cae_model'):
        run.cae_model = None
    if run.cae_model is None:
        auto_encoder = CAE(input_shape=run.input_shape, pool_size=[4, 4, 2], optimizer='adam') # Main
        run.cae_model = auto_encoder
        save_var(run, run.results_path + 'run.pkl', make_path=True)
        auto_encoder = auto_encoder.generate()
        auto_encoder.save(run.results_path + 'model_before_training.h5')
    else:
        weights_file = list_all_files(run.last_run, pattern='weights_epoch*')[-1]
        initial_epoch = int(weights_file.split('_')[-1].split('.')[0]) + 1
        try:
            auto_encoder = run.cae_model.generate()
        except Exception as e:
            red_print(f'Could not read the model, re initializing the model: {e}')
            auto_encoder = CAE(input_shape=run.input_shape, pool_size=[4, 4, 2], optimizer='adam')
            # auto_encoder = CAE(input_shape=run.input_shape, pool_size=[4, 4, 2], optimizer='RMSprop', metrics='mse')
            run.cae_model = auto_encoder
            auto_encoder = auto_encoder.generate()
        auto_encoder.load_weights(weights_file)

    # Check points
    callbacks_list = [
        ModelCheckpoint(run.results_path + 'checkpoint_model.hdf5', monitor='loss', verbose=1),
        LossHistory(run=run, acc_calc=True)
    ]

    cpu = get_arg_terminal('cpu', default=tot_cpu)

    print(f'Running on {cpu} CPUs.')
    print(f'Starting from epoch# {initial_epoch} to epoch# {run.num_epochs}')

    print('Right before fitting', flush=True)
    history = auto_encoder.fit_generator(
        generator=run.train_generator,
        steps_per_epoch=len(run.train_generator),
        epochs=run.num_epochs,
        verbose=1,
        shuffle=True,
        validation_data=run.test_generator,
        validation_steps=len(run.test_generator),
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=cpu,
        max_queue_size=10,
        initial_epoch=initial_epoch,
    )

    run.last_epoch = run.num_epochs

    print(f'Fitting {run.num_epochs} completed!', flush=True)

    auto_encoder.save(run.results_path + f'model_{run.num_epochs}.h5')
    auto_encoder.save_weights(run.results_path + f'weights_{run.num_epochs}.h5')

    # Extracting features
    print('Feature extraction')
    pos_x_train = feature_extraction(generator=run.train_generator, run=run, auto_encoder=auto_encoder)
    save_var(pos_x_train, run.results_path + 'pos_train_features.pkl', compress_np=True)
    del pos_x_train

    pos_x_test = feature_extraction(generator=run.test_generator, run=run, auto_encoder=auto_encoder)
    save_var(pos_x_test, run.results_path + 'pos_test_features.pkl', compress_np=True)
    del pos_x_test

    neg_x = feature_extraction(run.params['neg_x'], run=run, auto_encoder=auto_encoder, frac=0.2 / 200)
    save_var(neg_x, run.results_path + 'neg_features.pkl', compress_np=True)
    del neg_x

    whole_dataset_feature_extraction(run=run)

    save_var(history.history, run.results_path + f'history_{run.num_epochs}.pkl')
    run.end()


def add_session_to_plot(ses):
    run = ses['run']
    history = ses['history']
    D = len(run.model.input.shape) - 2
    loss = run.model.loss
    plt.plot(history.epoch, history.history['loss'], label='Loss: {}'.format(loss))
    plt.plot(history.epoch, history.history['val_loss'], label='Loss: {} (Evaluation)'.format(loss))
    plt.legend()


def plot_cae():
    tf_shut_up()

    for file in list_all_files('results/run_0*/', 'autoencoder*.pkl')[-4:]:
        print('File= ', file)
        session = load_var(file)
        run_var = RunSet(session['run'])
        plt.figure(0)
        plt.title('Crystal - CAE\nPad: 17.5 & bins: 32')
        add_session_to_plot(session)
        plt.figure()
        plt.title('Crystal - CAE\n Loss: {}'.format(run_var.model.loss))
        for key in run_var.history.history:
            if key == 'loss' or key == 'val_loss':
                plt.plot(run_var.history.epoch, run_var.history.history[key], label=key, linewidth=4)
            else:
                plt.plot(run_var.history.epoch, run_var.history.history[key], label=key)
        plt.legend()
    for i in plt.get_fignums():
        plt.figure(i)
        axes = plt.gca()
        axes.set_xlim([0, 50])
        axes.set_ylim([0, .8])
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.savefig('results/plot_{}.png'.format(i), dpi=600)
    plt.show()


if __name__ == '__main__':
    # run = RunSet(path='results/run_006/run.pkl')
    train_cae()
    # plot_cae()
    print('The End')
