# from utility.util_crystal import RunSet
import os
from keras.callbacks import Callback
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
# from utility.utility_general import *
# from xiespp.utility.utility_general import *

def tf_shut_up():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging, compat
        # logging.set_verbosity(logging.ERROR)
        compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func

            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


# class LossHistory(Callback):
#     def __init__(self, run: RunSet, filename='checkpoint_loss_history.pkl', acc_calc=False):
#         super().__init__()
#         self.losses = []
#         self.filename = filename
#         self.logs = []
#         self.x = []
#         self.val_losses = []
#         self.i = 0
#         self.run = run
#         self.time = datetime.now()
#         self.acc_calc = acc_calc
#         self.batch_loss = []
#         if acc_calc:
#             self.acc = []
#             self.val_acc = []
#
#     # def on_train_begin(self, logs=None):
#     #     self.losses = []
#
#     def on_batch_end(self, batch, logs=None):
#         tot_batch = len(self.run.train_generator)
#         t = str(datetime.now()).split('.')[0].split(' ')[1]
#         loss = logs['loss']
#         logs.update({'time': t})
#         self.logs.append(logs)
#         print(f'\nbatch: {batch}, time: {t}, loss: {loss}', flush=True)
#         freq = 50
#         if batch in np.floor(np.linspace(0, tot_batch, freq)):
#             print(f'Save point after passing {100/freq:.1f}% of batches')
#             self.model.save(self.run.results_path + 'model_tmp.h5')
#             self.model.save_weights(self.run.results_path + 'weights_tmp.h5')
#             save_df(pandas.DataFrame(self.logs), filename=self.run.results_path + 'batch_logs.txt')
#             gc.collect()
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.model.save(self.run.results_path + f'model_epoch_{epoch}.h5')
#         self.model.save_weights(self.run.results_path + f'weights_epoch_{epoch}.h5')
#         save_df(pandas.DataFrame(self.logs), filename=self.run.results_path + f'epoch_{epoch}_logs.txt')
#         # self.logs.append(logs)
#         self.x.append(self.i + 1)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.i += 1
#         if self.acc_calc:
#             self.acc.append(logs['acc'])
#             self.val_acc.append(logs['val_acc'])
#             plt.figure()
#             plt.plot(self.x, self.acc, label="acc")
#             plt.plot(self.x, self.val_acc, label="val_acc")
#             plt.legend()
#             plt.xlabel('Epochs')
#             plt.ylabel('Acc')
#             plt.savefig(self.run.results_path + 'plots_on_epoch_end_acc.png')
#             plt.close()
#         plt.figure()
#         plt.plot(self.x, self.losses, label="loss")
#         plt.plot(self.x, self.val_losses, label="val_loss")
#         plt.legend()
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         ax = plt.figure().gca()
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         plt.savefig(self.run.results_path + 'plots_on_epoch_end.png')
#         plt.savefig(self.run.results_path + 'plots_on_epoch_end.svg')
#         plt.close()
#         print('Evaluation time: {}\t'.format(datetime.now() - self.time), flush=True, end='')
#         print('Remaining time: {}'.format((datetime.now() - self.time) * (self.run.num_epochs - epoch)), flush=True)
#         self.time = datetime.now()


def random_seed(seed_value=0):
    # Seed value
    # Apparently you may use different seed values at each stage
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value

    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    tf.compat.v1.set_random_seed(seed_value)
    # tf.random.set_seed(random_seed)


def tf_select_gpu(gpu_n, verbose=True):
    physical_devices = tf.config.list_physical_devices('GPU')
    if verbose:
        print("Total Num. GPUs Available: ", len(physical_devices))
    try:
        # Restrict TensorFlow to only use the #n GPU
        tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        if verbose:
            print(len(physical_devices), "Physical GPUs,", len(logical_devices), "Logical GPU")
    except Exception as e:
        # Invalid device or cannot modify virtual devices once initialized.
        print('Exception thrown')
        print(str(e))
        pass

    if verbose:
        print('\nSelected resources:')
        for d in tf.config.get_visible_devices():
            print(d.name)


def tf_reset_gpu():
    from numba import cuda
    device = cuda.get_current_device()
    device.reset()


class BatchLogger(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.batches = []
        self.loss = []
        self.epochs = []
        self.epoch_number = None
        self.path = path

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_number = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])
        self.batches.append(batch)
        self.epochs.append(self.epoch_number)

    def on_epoch_end(self, epoch, logs=None):
        path = self.path
        if not self.path[-4:] == '.csv':
            path = path + '/batches.csv'
        import pandas as pd
        pd.DataFrame({'epoch': self.epochs, 'batch': self.batches, 'loss': self.loss})\
            .to_csv(path, index=False)


tf_shut_up()
