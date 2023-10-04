import CVR
# from utility.imports import *
from datetime import datetime
from pathlib import Path
import pandas as pd
import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from . import uplt as Plots
# from . import models

# noinspection PyBroadException
try:
    import tensorflow as tf
    # from . import tensorflow as utf
    from keras.utils import plot_model
except Exception:
    print("Could not import tensorflow.")
    tf = None


def get_callbacks(result_dir, batch_logger=True):
    from keras import callbacks
    
    # from keras.callbacks import CSVLogger
    # from keras.callbacks import EarlyStopping, ModelCheckpoint
    
    model_checkpoint = callbacks.ModelCheckpoint(
        result_dir / 'weights' / 'weights{epoch:04d}.h5',
        # monitor='val_acc',
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    # early_stop = callbacks.EarlyStopping(monitor='val_loss',
    #                            min_delta=0,
    #                            patience=15,
    #                            verbose=1,
    #                            mode='auto')
    csv_logger = callbacks.CSVLogger(result_dir / 'training.log',
                                     separator=',',
                                     append=True)
    # if batch_logger:
    #     batch_logger = utf.BatchLogger(result_dir / 'training-batch.log')
    # else:
    #     batch_logger = None

    callback_list = [
        # early_stop,
        model_checkpoint,
        csv_logger,
        batch_logger
    ]
    callback_list = [c for c in callback_list if c is not None]
    return callback_list


def train(model, gens, params, train_sub_dir="training", **kwargs):
    time_1 = datetime.now()
    result_dir_training = params["result_dir"] / train_sub_dir
    result_dir_training.mkdir(exist_ok=True)
    (result_dir_training/'weights').mkdir(exist_ok=True)

    callbacks = get_callbacks(
        result_dir=result_dir_training,
        batch_logger=params.get("batch_logger", False),
    )

    with tf.device(params["gpu_device"]):
        gens["train"].training = True
        history = model.fit(
            gens["train"],
            # epochs=params['epochs'],
            epochs=kwargs.pop("epochs", params["epochs"]),
            validation_data=gens["dev"],
            callbacks=callbacks,
            verbose=1,
            **kwargs,
        )
        gens["train"].training = False
    time_2 = datetime.now()

    df = pd.DataFrame(model.history.history)
    df.to_csv(result_dir_training / "history.csv", index=False)

    try:
        plot_model(model, show_shapes=True, to_file=result_dir_training / "model.png", dpi=200)
    except Exception:
        print("Could not plot model.")
        pass
    model.save(result_dir_training / "model.h5")
    if train_sub_dir == "training":
        model.save(params["result_dir"] / "model.h5")    

    print(f"Training time: {str(time_2 - time_1)[:-7]}")
    return history


def load_model(result_dir,
               model=None,
               weights=None,
               weights_path=None,
               weights_pick_method='best',
               weights_pick_metric='val_loss',
               weights_pick_mode='min',
               df=None,
               **kwargs):
    """
    Load model and weights from result_dir.
    :param result_dir:
    :param model:
    :param weights:
        True: load weights.h5
        'path/to/weights.h5': load weights from this file
    :param weights_path:
        'path/to/weights': load weights from this directory
    :param weights_pick_method:
        'best': load the weights with the best validation accuracy
        'last': load the last weights
    :param df:
        if weights_pick_method is 'best', df is required to find the best epoch
    :param kwargs:
    :return:
    """
    if model is None:
        model = result_dir/'model.h5'
    if weights_path is None:
        weights_path = 'training/weights'
    if isinstance(model, (str, Path)):
        from keras.models import load_model
        model = load_model(model, **kwargs)

    if weights_pick_method == 'best':
        if df is None:
            df = pd.read_csv(result_dir/weights_path/'history.csv')
        # best_epoch = df['val_acc'].idxmax()
        if weights_pick_mode == 'min':
            best_epoch = df[weights_pick_metric].idxmin()
        elif weights_pick_mode == 'max':
            best_epoch = df[weights_pick_metric].idxmax()
        weights = result_dir/weights_path/f'weights{best_epoch+1:04d}.h5'
    if weights_pick_method == 'last':
        # listing all the weights files
        weights_files = sorted((result_dir/weights_path).glob('weights*.h5'))
        if len(weights_files) == 0:
            raise FileNotFoundError(f'No weights files found in {result_dir/weights_path}')
        weights = weights_files[-1]

    if weights is True:
        weights = result_dir/'weights.h5'
    if isinstance(weights, (str, Path)):
        model.load_weights(weights)

    return model


def evaluation(model, gens, params, ensemble):
    result_dir = params['result_dir']
    ensemble_dir = result_dir / f'predictions/ensemble_{ensemble}'
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    for k, gen in gens.items():
        print(f'Evaluating {k} set with ensemble {ensemble}')
        predict(model, gen, params, ensemble=ensemble)
        df = gen.get_df()
        if 'ypl' in df.columns: 
            gen.info['acc'] = accuracy_score(df['y'], df['ypl'])
            print(f'Accuracy: {gen.info["acc"] * 100:.2f}%')
        cols_to_select = ['material_id', 'id', 'type', 'db', gen.y, 'yp', 'ypl', 'file']
        cols_to_select = [c for c in cols_to_select if c in df.columns]

        df[cols_to_select].rename(columns={gen.y: 'y'}).to_csv(ensemble_dir / f'{k}.csv')


def predict(model, gen, params=None, ensemble=None, compute_ypl=False, **kwargs):
    # with tf.device(params['gpu_device']):
    #     yp = model.predict(gen, verbose=1)
    # gen.df.loc[gen.index, 'yp'] = yp
    if params is None:
        params = {}
    params.update(kwargs)
    yp = CVR.keras_tools.predict(model=model, generator=gen, params=params, ensemble=ensemble, **kwargs)
    if compute_ypl or 'ypl' in params:
        gen.df['ypl'] = compute_labels(gen.df['yp'])
    return yp


def compute_labels(yp, threshold=0.5):
    ypl = (yp > threshold)
    ypl[np.isnan(yp)] = np.nan
    return ypl


def read_evaluations(result_dir, metrics=None, print_results=False):
    ensemble_dirs = list((result_dir / 'predictions').glob('ensemble_*'))
    evals = {ensemble_dir.stem: {Path(f).stem: dict(file=f)
                                 for f in list(ensemble_dir.glob('*.csv'))}
             for ensemble_dir in ensemble_dirs}

    for ensemble, files in evals.items():
        for k in files:
            df = pd.read_csv(evals[ensemble][k]['file'], index_col=0)
            evals[ensemble][k]['df'] = df
            evals[ensemble][k]['metrics'] = {}
            if 'ypl' in df.columns:
                evals[ensemble][k]['metrics']['acc'] = accuracy_score(df['y'], df['ypl'])
            if metrics is not None:
                import tensorflow as tf
                for metric_name in metrics:
                    metric_function = getattr(tf.keras.metrics, metric_name)
                    value = metric_function(df['y'].to_numpy(), df['yp'].to_numpy())
                    evals[ensemble][k]['metrics'][metric_name] = float(value)

    if print_results:
        for ensemble in evals:
            print(f'Ensemble {ensemble}')
            for set_name in evals[ensemble]:
                print(f'Set {set_name}:')
                for metric_name in metrics:
                    if metric_name in evals[ensemble][set_name]["metrics"]:
                        print(f'{metric_name}: {evals[ensemble][set_name]["metrics"][metric_name]:.4f}')
            print()


    return evals

def accuracy_score(y, yp):
    return pd.Series(y == yp).mean()

def plot_loss(result_dir=None, evals=None, history=None,
              # multi_evals=None
              ):
    if history is None:
        history = pd.read_csv(result_dir / 'training/history.csv')
    hist = history
    plot_acc = False
    if 'val_accuracy' in hist.columns:
        plot_acc = True

    fig_size = (11 / 2, 4)
    if plot_acc:
        fig_size = (11, 4)
    fig = plt.figure(figsize=fig_size)
    if plot_acc:
        plt.subplot(1, 2, 1)
    plt.plot(hist["loss"], label='Train')
    plt.plot(hist["val_loss"], label='Val')
    plt.legend()
    # plt.legend(["Train", "Val"])
    plt.title("Loss")

    if plot_acc:
        plt.subplot(1, 2, 2)
        plt.plot(hist["accuracy"], label='Train')
        plt.plot(hist["val_accuracy"], label='Val')

        # if multi_evals:
        #     evals = {f'{ensemble_name}-{k}': e for ensemble_name, ensemble in multi_evals.items()
        #              for k, e in ensemble.items() if k != 'case_study'}

        if evals is not None:
            for k, e in evals.items():
                if k == 'case_study':
                    continue
                plt.plot(len(hist["accuracy"]) - 1, e['acc'], marker='*', markersize=10, label=f'{k} Acc')
        plt.legend()
        # plt.legend(["Train", "Val"])
        plt.title("Accuracy")

    # plt.savefig(result_dir / 'training_history.png')
    # plt.show()
    return fig


def plot_loss_batch(result_dir=None, history=None):
    if history is None:
        history = pd.read_csv(result_dir / 'training/training-batch.log')
    hist = history
    fig = plt.figure(figsize=(11, 4))
    # plt.subplot(1, 2, 1)
    plt.plot(hist["loss"])
    plt.title("Train Set Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    # plt.show()
    return fig


def parity_plot(df: typing.Dict[str, pd.DataFrame] = None,
                y_col='y',
                yp_col='yp',
                y_label=None,
                hue=None,
                prettify_labels=True,
                ax=None,
                ):
    assert df is not None, 'df is None'
    # df_dict = df
    # if isinstance(df, pd.DataFrame):
    #     df_dict = {'Data': df}

    assert y_label is not None, 'y_label is None'
    if prettify_labels:
        y_label = y_label.replace('_', ' ').title()

    # mse = tf.keras.losses.MeanSquaredError()
    # mae = tf.keras.losses.MeanAbsoluteError()

    mae_per_set = (df[y_col] - df[yp_col]).abs().groupby(df[hue]).mean()
    df['mae'] = df[hue].map(lambda x: f'{x}: MAE = {mae_per_set[x]:.4f}')

    ax = sns.scatterplot(data=df, x=y_col, y=yp_col, ax=ax, alpha=0.3, hue='mae')
    df.drop(columns=['mae'], inplace=True)

    ax.plot(ax.get_xlim(), ax.get_xlim(), 'r', alpha=0.5)
    ax.set_ylabel(f'{y_label} Predicted'.capitalize())
    ax.set_xlabel(f'{y_label} Actual'.capitalize())
    # ax.set_title(f'{set_label} Set Parity'.title())

    # er = (df[y_col] - df[yp_col]).abs().mean()
    # Plots.put_costume_text_instead_of_legends(ax=ax, labels=f'MAE = {er:.4f}')

    # outputs[set_label] = {'ax': ax, 'mae': er}

    return ax
