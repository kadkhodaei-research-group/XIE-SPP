import pandas as pd
from pathlib import Path
import os, sys
from datetime import datetime
import numpy as np

# from utility.imports import *
# import utility.general as utg
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)  # To mute Tensorflow warnings
import uplt as Plots
import utg
import matplotlib.pyplot as plt
import seaborn as sns
import CVR
import typing

try:
    # import utility.tensorflow as util_tf
    import tensorflow as tf
    # from utility.util_image_generator_keras import *
except ModuleNotFoundError:
    # ImageGeneratorKeras = None
    pass

# from CVR.generator import *

# try:
#     from utility import util_torch
# except ModuleNotFoundError:
#     pass


# utg.pd_data_frame_display_setting()


class ModelTrainer:
    def __init__(self, result_dir=None, gpu_n=None, random_seed=0, framework='tf', **kwargs):
        if result_dir is None:
            result_dir = os.path.join('../results', 'test')
        input_params = locals()

        # if (gpu_n == -2) or (gpu_n == 'auto'):
        #     gpu_n = utg.get_free_er_gpu()

        self.result_dir = Path(result_dir)
        self.gpu_n = gpu_n
        # self.n_jobs = kwargs.get('n_jobs', utg.tot_cpu)
        self.n_jobs = kwargs.get('n_jobs', 8)
        self.gpu_device = f'/device:GPU:{str(gpu_n)}'
        self.random_seed = random_seed
        self.tot_time_1 = None
        self.tot_time_2 = None
        self.framework = framework
        self.params = kwargs
        # self.image_params =

        # self.prepare_system()

    def prepare_system(self):
        # locals().update(self.params)
        self.prepare_gpu(gpu_n=self.gpu_n)
        os.makedirs(self.result_dir, exist_ok=True)
        self.tot_time_1 = datetime.now()
        # self.params['git'] = git_show_version(verbose=False)
        # utg.set_random_seed(self.random_seed)
        # utg.save_var(locals(), str(Path(self.result_dir, 'input_params.pkl')), make_path=True)

        print(f'{"Executable": <15}: ', sys.executable)
        print(f'{"Results": <15}: ', self.result_dir)
        return self

    def prepare_gpu(self, gpu_n='auto'):
        if gpu_n == 'auto':
            # gpu_n = utg.get_free_er_gpu()
            assert False, 'Auto GPU selection is not implemented yet!'

        # if self.framework == 'tf':
        #     self.params['device'] = util_tf.tf_select_gpu(gpu_n)
        # if self.framework == 'torch':
        #     self.params['device'] = util_torch.select_device(gpu_n=gpu_n)

        self.gpu_n = gpu_n
        self.gpu_device = f'/device:GPU:{str(gpu_n)}'

    def prepare_data(self, data):
        # if not utg.is_list_like(data):
        #     data = [data]
        generator = DataPreparation(
            data={'data': data},
            box=self.params['box'],
            filling=self.params['filling'],
            channels=self.params['channels'],
        ).prepare()
        return generator

    def train_model(self, model, train_generator, test_generator, callbacks=None,
                    verbose=1):
        # assert util_tf.check_model_compatibility_with_dataset(model, train_generator)
        assert self.tot_time_1 is not None, 'Please call prepare_system() first!'

        if callbacks is None:
            from formation_energy import models_prep
            callbacks = models_prep.generate_callbacks_list(self.result_dir)

        time_1 = datetime.now()

        with tf.device(self.gpu_device):
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=self.params['epochs'],
                validation_data=test_generator,
                validation_steps=len(test_generator),
                callbacks=callbacks,
                verbose=verbose,
                #         use_multiprocessing=True,
                workers=32,
                max_queue_size=20,
            )

        time_2 = datetime.now()
        print('End of the training. Run time:', str(time_2 - time_1)[:-7])

        try:
            model.save(self.result_dir / 'model.h5')
        except Exception as e:
            print('Model could not be saved.')
            print(str(e))
        hist = pd.DataFrame(history.history)
        hist = hist.reset_index().rename(columns={'index': 'epoch'})
        hist['epoch'] += 1
        hist.to_csv(self.result_dir / 'history.csv', index=False)
        pass

    def load_model(self, epoch=None, weights_file=None, model=None):
        if (epoch is None) and (weights_file is None):
            if Path(self.result_dir / 'weights-picked.h5').exists():
                weights_file = self.result_dir / 'weights-picked.h5'
            else:
                # epoch = len(utg.list_all_files(Path(self.result_dir, 'weights'), pattern='*'))
                epoch = len(list((Path(self.result_dir) / 'weights').glob('*')))

        if weights_file is None:
            weights_file = Path(self.result_dir, 'weights', f'weights{epoch:04}.h5')
        loaded_model = model
        if model is None:
            from keras.models import load_model
            loaded_model = load_model(Path(self.result_dir, 'model.h5'))
        loaded_model.load_weights(weights_file)

        return loaded_model

    def predict(self, model, generator: CVR.ImageGeneratorKeras, device=None, include_nan=False, ensemble=None,
                return_all_ensembles=False):
        # from .model import predict
        from .run import predict
        yp = predict(
            model=model, gen=generator,
            device=device or self.gpu_device,
            include_nan=include_nan,
            ensemble=ensemble,
            return_all_ensembles=return_all_ensembles
        )
        return yp

    def evaluate_model(self, model, train_generator: CVR.ImageGeneratorKeras, test_generator: CVR.ImageGeneratorKeras):
        time_1 = datetime.now()
        with tf.device(self.gpu_device):
            yp_train = model.predict(
                train_generator,
                batch_size=len(train_generator),
                verbose=1,
                #         use_multiprocessing=True,
                workers=32,
                max_queue_size=20,
            )
            yp_test = model.predict(
                test_generator,
                batch_size=len(test_generator),
                verbose=1,
                #         use_multiprocessing=True,
                workers=32,
                max_queue_size=20,
            )

        y_train = train_generator.get_labels()
        y_test = test_generator.get_labels()

        id_train = [im.image_id for _, im in train_generator.df['image'].iteritems()]
        id_test = [im.image_id for _, im in test_generator.df['image'].iteritems()]

        time_2 = datetime.now()
        print('End of evaluations. Run time:', str(time_2 - time_1)[:-7])

        df_train = pd.DataFrame({'id': id_train, 'y': y_train.flatten(), 'yp': yp_train.flatten()})
        df_test = pd.DataFrame({'id': id_test, 'y': y_test.flatten(), 'yp': yp_test.flatten()})

        df_train.to_csv(Path(self.result_dir, 'df_train.csv'), index=False)
        df_test.to_csv(Path(self.result_dir, 'df_test.csv'), index=False)
        return {'df_train': df_train, 'df_test': df_test}

    def evaluate(self, model, generators: typing.List[CVR.ImageGeneratorKeras], save_csv=True, ensemble=None):
        time_1 = datetime.now()

        if isinstance(generators, CVR.ImageGeneratorKeras):
            generators = [generators]
        evaluations = {}
        for gen in generators:
            print(f'Predicting {gen.label} ...', flush=True)
            yp = self.predict(model, gen, ensemble=ensemble, include_nan=False, return_all_ensembles=False)

            # y = gen.get_labels()
            # if 'id' in gen.df.columns:
            #     sample_id = gen.df['id']
            # else:
            #     sample_id = [im.image_id for _, im in gen.df['image'].iteritems()]
            # df = pd.DataFrame({'id': sample_id, 'y': y.flatten(), 'yp': yp.flatten()})
            # df = pd.DataFrame({'id': sample_id, 'y': y.flatten(), 'yp': yp.mean(axis=1)})
            # df = pd.concat([df, pd.DataFrame(yp, index=df.index)], axis=1)
            df = gen.get_df()[['id', 'y']].copy()
            df['yp'] = yp
            if save_csv:
                df.to_csv(Path(self.result_dir, f'df_{gen.label}.csv'), index=False)
            evaluations[gen.label] = df

        time_2 = datetime.now()
        print('End of evaluations. Run time:', str(time_2 - time_1)[:-7])
        return evaluations

    def get_results(self, details=None) -> pd.DataFrame:
        dfs = utg.list_all_files(self.result_dir, pattern='df_*.csv', )
        dfs = {Path(x).stem.replace('df_', ''): pd.read_csv(x) for x in dfs}
        for k in dfs:
            dfs[k]['set'] = k
            # dfs[k]['error'] = np.abs(dfs[k]['y'] - dfs[k]['yp'])
            from sklearn import metrics
            dfs[k]['mae'] = metrics.mean_absolute_error(dfs[k]['y'], dfs[k]['yp'])
            dfs[k]['mse'] = metrics.mean_squared_error(dfs[k]['y'], dfs[k]['yp'])
            # dfs[k]['Sets'] = f'{k.capitalize()} - MAE={int(np.round(dfs[k]["mae"].values[0]*1000, 0)):,d}'
            dfs[k]['Sets'] = f'{k.capitalize()} - MAE={np.round(dfs[k]["mae"].values[0], 3):.3f}'

        df_mixed = pd.concat([dfs[x] for x in dfs], axis=0)
        # dfs['mixed'] = df_mixed
        dfs = df_mixed

        if details:
            if not isinstance(details, str):
                details = Path('../Data/data_banks/df.csv')
            df_details = pd.read_csv(details)
            # right_on = 'material_id'
            # if 'material_id' not in df_details.columns:
            #     right_on = 'id'
            dfs = pd.merge(dfs, df_details,
                           left_on='material_id' if 'material_id' in dfs.columns else 'id',
                           right_on='material_id' if 'material_id' in df_details.columns else 'id',
                           how='left')
        return dfs

    def plot_loss(self, history=None):
        if history is None:
            history = pd.read_csv(os.path.join(self.result_dir, 'history.csv'))
            # history = pd.read_csv(self.result_dir/ 'training.log')
        metric_columns = [x for x in history.columns if 'loss' not in x and 'epoch' not in x]

        # plt.figure(figsize=(11, 4))
        fig, axs, fs = Plots.plot_format(ncols=2, size=350, font_size=12)
        # plt.subplot(1, 2, 1)
        ax = axs[0]
        ax.plot(history["loss"])
        ax.plot(history["val_loss"])
        ax.legend(["Train", "Validation"])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("loss")
        ax.set_title("loss fn.")
        ax = axs[1]
        ax.plot(history[metric_columns[0]])
        ax.plot(history["val_" + metric_columns[0]])
        ax.legend(["Train", "Validation"])
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric_columns[0])
        ax.set_title(metric_columns[0])
        Plots.plot_save('loss', self.result_dir)

        return fig, axs

    def plot_parity_plots(self, df=None, save_plots=True, prettify_labels=True):
        if df is None:
            df = {'df_train': pd.read_csv(Path(self.result_dir, 'df_train.csv')),
                  'df_test': pd.read_csv(Path(self.result_dir, 'df_test.csv'))}

        hist = pd.read_csv(Path(self.result_dir) / 'history.csv') if (
                Path(self.result_dir) / 'history.csv').exists() else None
        df_train = df['df_train']
        df_validation = df['df_test']
        y_label = self.params['y']
        if prettify_labels:
            y_label = y_label.replace('_', ' ').title()

        mse = tf.keras.losses.MeanSquaredError()
        mae = tf.keras.losses.MeanAbsoluteError()

        y = df_train
        mse_train = mse(y["y"], y["yp"]).numpy()
        mae_train = mae(y["y"], y["yp"]).numpy()

        y = df_validation
        mse_validation = mse(y["y"], y["yp"]).numpy()
        mae_validation = mae(y["y"], y["yp"]).numpy()
        print(f'Train MAE loss: {mae_train:.5f}')
        print(f'Validation MAE loss: {mae_validation:.5f}')
        print(f'Evaluations and training history diff.: \
            {(mae_train - hist["mean_absolute_error"].iloc[-1]) / mae_train * 100:.1f}%')

        sets = {'Train': {}, 'Validation': {}}
        for set_label, df in zip(list(sets.keys()), [df_train, df_validation]):
            f, ax, fs = Plots.plot_format(equal_axis=True, size=100, font_size=12)
            sns.scatterplot(data=df, x='y', y='yp', ax=ax, alpha=0.3)
            ax.plot(ax.get_xlim(), ax.get_xlim(), 'r', alpha=0.5)
            ax.set_ylabel(f'{y_label} Predicted')
            ax.set_xlabel(f'{y_label}')
            ax.set_title(f'{set_label} Set Parity')

            er = locals()[f'mae_{set_label.lower()}']
            Plots.put_costume_text_instead_of_legends(ax=ax, labels=f'MAE = {er:.4f}')

            sets[set_label]['ax'] = ax
            sets[set_label]['f'] = f
            if save_plots:
                Plots.plot_save(f'{set_label.lower()}_set_parity', self.result_dir)
                plt.show()

        return sets

    def parity_plot(self,
                    df: typing.Dict[str, pd.DataFrame] = None,
                    y_col='y',
                    yp_col='yp',
                    y_label=None,
                    save_plots=True,
                    show_plots=True,
                    prettify_labels=True,
                    ):
        assert df is not None, 'df is None'
        df_dict = df
        if isinstance(df, pd.DataFrame):
            df_dict = {'df': df}

        y_label = y_label or self.params.get('y')
        assert y_label is not None, 'y_label is None'
        if prettify_labels:
            y_label = y_label.replace('_', ' ').title()

        # mse = tf.keras.losses.MeanSquaredError()
        mae = tf.keras.losses.MeanAbsoluteError()

        outputs = {}

        for set_label, df in df_dict.items():
            f, ax, fs = Plots.plot_format(equal_axis=True, size=100, font_size=12)
            sns.scatterplot(data=df, x=y_col, y=yp_col, ax=ax, alpha=0.3)
            ax.plot(ax.get_xlim(), ax.get_xlim(), 'r', alpha=0.5)
            ax.set_ylabel(f'{y_label} Predicted'.capitalize())
            ax.set_xlabel(f'{y_label}'.capitalize())
            ax.set_title(f'{set_label} Set Parity'.title())

            er = mae(df[y_col], df[yp_col]).numpy()
            Plots.put_costume_text_instead_of_legends(ax=ax, labels=f'MAE = {er:.4f}')

            if save_plots:
                Plots.plot_save(f'{set_label.lower()}_set_parity', self.result_dir)
            if show_plots:
                plt.show()

            outputs[set_label] = {'ax': ax, 'mae': er}

        return outputs

    def end_training(self):
        self.tot_time_2 = datetime.now()
        print('End of the training. Run time:', str(self.tot_time_2 - self.tot_time_1)[:-7], '\n\n')

        # print('Releasing the allocated GPU(s)')
        # utg.release_gpu()  # Releasing allocated GPUs

        # print('Storing the *.py files in a tar file.')
        # py_files = utg.list_all_files('..', pattern='*.py', recursive=False)
        # py_files += utg.list_all_files('../utility', pattern='*.py', recursive=False)
        # utg.make_tar_file(py_files, 'codes.tar', self.result_dir)

        # if utg.is_jupyter_notebook():
        #     print('Saving the notebook')
        #     utg.save_jupyter_notebook_to_dir(self.params['notebook_filename'], self.result_dir)


class DataPreparation:
    def __init__(self,
                 data,
                 box: CVR.BoxImage = None,
                 filling: str = None,
                 channels: typing.List[str] = None,
                 y: str = None,
                 random_rotation: bool = True,
                 batch_size: int = 32,
                 **kwargs,
                 ):
        # self.data = data
        self.box = box
        self.filling = filling
        self.channels = channels
        self.y = y
        self.random_rotation = random_rotation
        self.batch_size = batch_size

        self.input_shape = tuple([self.box.n_bins] * 3 + [len(self.channels)])

        self.datasets = data
        self.params = kwargs.get('data_preparation_params', {})

    def prepare(self):
        # import swifter
        # print(f'Pandas parallelization with swifter version: {swifter.__version__}')

        t1 = datetime.now()

        # if utg.is_list_like(self.datasets):
        #     self.datasets = {'Data': self.datasets}
        for key, data_set in self.datasets.items():
            if not isinstance(data_set, pd.DataFrame):
                self.datasets[key] = pd.DataFrame({'atoms': list(data_set)})
                if hasattr(data_set[0], 'info'):
                    if 'material_id' in data_set[0].info:
                        self.datasets[key]['material_id'] = [r.info['material_id'] for r in data_set]
            if self.y is None:
                self.y = 'property'
                self.datasets[key][self.y] = 0

        self.box.make_rotational_boxes()
        generators = {}

        for key, data_set in self.datasets.items():
            # Preparing the image classes
            print('Dataset:', key)
            # print('Preparing the image classes ...', flush=True)
            img = [CVR.ThreeDImage(atoms=r['atoms'], image_id=r['material_id'], **self.__dict__) for _, r in
                   data_set.iterrows()]
            # pbar(data_set.iterrows(), total=len(data_set))]
            data_set.loc[:, 'image'] = img

            print('Checking the image requirements', flush=True)
            # data_set['req. check'] = data_set['image'].swifter.apply(lambda x: x.check_requirements())
            data_set.loc[:, 'is_valid'] = utg.parallel_apply(
                data_set.loc[:, 'image'], lambda x: x.check_requirements(),
            )

            print(f'{np.count_nonzero(~data_set["is_valid"])} images did not meet the requirements'
                  f' and were removed from the {key} set.')
            ind_valid = data_set[data_set['is_valid']].index

            print(f'Preparing the point clouds for data set: {key}', flush=True)
            data_set['point_cloud'] = None
            tmp = utg.parallel_apply(
                data_set.loc[ind_valid, 'image'],
                lambda x, random_rotation: x.get_point_cloud(random_rotation=random_rotation),
                random_rotation=self.random_rotation,
            )
            for n, i in enumerate(ind_valid):
                data_set.loc[i, 'image'].set_point_cloud(tmp[n])
            print(f'Point clouds for data set: {key} are ready')

            print('Preparing the image generators ...', flush=True)
            data_set['y'] = None
            if self.y is not None:
                data_set['y'] = data_set[self.y]
            data_set['id'] = None
            if 'material_id' in data_set:
                data_set['id'] = data_set['material_id']
            # if keep_all_data:
            generators[key] = CVR.ImageGeneratorKeras(
                data_set[['id', 'image', self.y, 'y', 'is_valid']],
                label=f'{key}',
                **self.__dict__,
            )
            # else:
            #     generators[key] = ImageGeneratorKeras(
            #         data_set.loc[ind_valid, ['id', 'image', self.y, 'y']],
            #         label=f'{key}',
            #         **self.__dict__,
            #     )
            print(f'{"Stats": <15}: ', end='')
            generators[key].stats()

        t2 = datetime.now()
        print(f'Time to prepare the image generators: {str(t2 - t1)[:-7]}')
        return generators


def get_model_trainer(path: str = None):
    if path is None:
        path = '../results/train_improvement_9/train_15lm-6sk-t1(model7-all-mp)'
    model_trainer = ModelTrainer(
        result_dir=path,
        channels=['atomic_number', 'group', 'period'],
        box=CVR.BoxImage(box_size=70 // 4, n_bins=128 // 4),
        filling='fill-cut',
        y='formation_energy_per_atom',
        random_rotation=True,
        batch_size=32,
    )
    # model_trainer.gpu_device = '/device:CPU:0'
    return model_trainer


def get_results(path: str = None, details=True):
    mt = get_model_trainer(path)
    return mt.get_results(details=details)


if __name__ == '__main__':
    # result_dir = os.path.join('results', 'train_improvement_5')
    # gpu_n = 1
    # epochs = 30
    # random_seed = 0
    # box = 70
    # n_bins = 128
    # channels = ['atomic_number', 'group', 'period']
    # filling = 'fill-cut'
    #
    # trainer = ModelTrainer(
    #     result_dir=result_dir,
    #     gpu_n=gpu_n,
    #     random_seed=random_seed,
    #     epochs=epochs,
    #     box=box,
    #     n_bins=n_bins,
    #     channels=channels,
    #     filling=filling,
    # )
    # trainer.prepare_system()
    # git_update()
    get_model_trainer().load_model(epoch=467)
    pass
