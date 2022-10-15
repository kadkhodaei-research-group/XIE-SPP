import pandas as pd
import pickle
import sys, os

org_dir = os.getcwd()
rep_dir = '.'


def synthesizability_predictor(data, classifier='cae-mlp', verbose=1, use_multiprocessing=False, workers=1):
    """
    :param data: a list of cif files or ASE Atoms objects for evaluation
    :param classifier: 'cae-mlp' or 'cnn'
    :param verbose: mode 0 or 1
    :param use_multiprocessing: Boolean
    :param workers: Integer. Maximum number of processors. -1 to use all
    :return: the synthesizability likelihood of the input list
    """

    def cnn():
        try:
            trained_model = load_model(rep_dir + '/' + config.cnn_model_dir + '/model.h5')
        except Exception:
            from keras.models import Sequential
            from keras.layers import Input, Conv3D, Flatten, Dense, MaxPooling3D, Dropout
            from keras.regularizers import l2
            from keras.optimizers import Adam
            from keras.callbacks import EarlyStopping, ModelCheckpoint
            from keras.utils import Sequence
            import keras.backend as K
            from keras import layers

            input_shape = (128, 128, 128, 3)
            mlp_nodes_p_layer = 13
            auto_encoder = models.CAE(input_shape=input_shape, pool_size=[4, 4, 4], optimizer='adam',
                                      metrics=['accuracy'], channels=[32, 32, 32], verbose=False)
            auto_encoder = auto_encoder.generate()
            cae_1 = auto_encoder
            for i in range(len(cae_1.layers)):
                if ('encode' in cae_1.layers[-1].name) or ('pooling' in cae_1.layers[-1].name):
                    break
                cae_1.pop()
            cae_1.add(layers.Flatten())
            cae_1.add(layers.Dense(mlp_nodes_p_layer))
            cae_1.add(layers.Dense(mlp_nodes_p_layer))
            cae_1.add(layers.Dense(mlp_nodes_p_layer))
            cae_1.add(layers.Dense(1, activation='sigmoid'))
            trained_model = cae_1

        trained_model.load_weights(rep_dir + '/' + config.cnn_model_dir + '/weights0006.h5')

        yp = trained_model.predict(
            generator,
            steps=len(generator),
            verbose=verbose,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            max_queue_size=10,
        )
        yp = yp.flatten()
        return yp

    def cae_mlp():
        try:
            encoder = load_model(rep_dir + '/' + config.cae_mlp_model_dir + '/encoder.h5', compile=False)
        except Exception:

            from keras.models import Sequential
            from keras.layers import Input, Conv3D, Flatten, Dense, MaxPooling3D, Dropout
            from keras.regularizers import l2
            from keras.optimizers import Adam
            from keras.callbacks import EarlyStopping, ModelCheckpoint
            from keras.utils import Sequence
            import keras.backend as K
            import models
            from keras import layers

            lr = 0.0005
            lr_decay = 0.002
            opti = Adam(lr=lr, decay=lr_decay)

            input_shape = (128, 128, 128, 3)
            auto_encoder = models.CAE(input_shape=input_shape, pool_size=[4, 4, 2], optimizer=opti,
                                      dropout_rate=0.3, verbose=False)
            trained_cae = auto_encoder.generate()
            for i in range(len(trained_cae.layers)):
                if ('encode' in trained_cae.layers[-1].name) or ('pooling' in trained_cae.layers[-1].name):
                    break
                trained_cae.pop()
            trained_cae.add(layers.Flatten())
            encoder = trained_cae

        encoder.load_weights(rep_dir + '/' + config.cae_mlp_model_dir + '/encoder_weights.h5')
        with open(rep_dir + '/' + config.cae_mlp_model_clf_dir + '/classifier_class.pkl', 'rb') as f:
            clf = pickle.load(f)
        lsr = encoder.predict(
            generator,
            steps=len(generator),
            verbose=verbose,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            max_queue_size=10,
        )
        yp = clf.predict_proba(lsr)
        return yp

    # sys.path.append(repository_path)
    tf_mutter()
    os.chdir(rep_dir)
    import image_generator
    import config
    import models
    import ml_tools, crystals_tools
    os.chdir(org_dir)
    from tensorflow.keras.models import load_model

    target_col = 'img_path'
    df = pd.DataFrame(data=list(data), columns=[target_col])
    generator = image_generator.ImageGeneratorDataFrame(df)
    out = None
    if classifier == 'cnn':
        out = cnn()
    if (classifier == 'cae-mlp') or (classifier == 'cae_mlp'):
        out = cae_mlp()

    return out


def tf_mutter():
    try:
        import os
        from tensorflow import logging, compat
        compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper
        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated
    except ImportError:
        pass


def get_test_samples(samples='GaN'):
    import glob
    return glob.glob(rep_dir + f'/finalized_results/explore_structures/cif/{samples}/*.cif')

rep_dir = './'
# from importlib import resources
# rep_dir = resources.path(package=synthesizabliity, resource="").__enter__()
# rep_dir = "/Users/ali/GitHub/XIE-SPP"


if __name__ == '__main__':
    import argparse
    # classifier = 'cae-mlp', verbose = 1, use_multiprocessing = False, workers = 1
    parser = argparse.ArgumentParser(description='Crystal synthsizability predicttor')
    # parser.add_argument('--test', action='store_true', help='Run on test samples')
    parser.add_argument('-f', '--file', type=str, help='CIF file to predict')
    parser.add_argument('-c', '--classifier', type=str, default='cae-mlp', help='Classifier to use (cnn or cae-mlp). Default is cae-mlp')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level. Default is 0')
    parser.add_argument('-m', '--multiprocessing', action='store_true', help='Use multiprocessing. Default is False')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of workers. Default is 1')
    parser.add_argument('-o', '--output', type=str, help='Output file name.')

    args = parser.parse_args()

    files = None
    if args.test:
        files = get_test_samples()
        if len(files) == 0:
            print('No test samples found.')
            exit(1)
    if args.file:
        files = [args.file]
    if files is None or len(files) == 0:
        print('No CIF file was provided. Use -f or --file to provide a CIF file.')
        exit(1)

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)  # To mute Tensorflow warnings

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = synthesizability_predictor(
            files,
            classifier=args.classifier,
            verbose=args.verbose,
            use_multiprocessing=args.multiprocessing,
            workers=args.workers
        )
    output = list(output)
    print(output)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(str(output))
    pass
