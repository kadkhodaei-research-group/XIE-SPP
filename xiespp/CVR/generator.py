# from util_cvr import *
from . import util_cvr as utg
# import utility.utility_general as utg
import numpy as np
import pandas as pd
from .image_processor import ImagePreprocessor, prepare_df

# from utility import crystal_image_tools, utility_crystal

# noinspection PyBroadException
try:
    # noinspection PyUnresolvedReferences
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)  # To mute Tensorflow warnings
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.utils import Sequence
    # import tensorflow.keras.utils.Sequence as Sequence
except Exception:
    Sequence = object
    pass


class ImageGenerator:
    """
    A data (3d image) generator object used for passing a lot of data to Keras models.
    Args:
        data: A data frame including the image objects and their labels
            Can be also created from a list of crystal files - ASE atomic objects - ThreeDImages -
            PyMatGen Structure objects also CVR ImageGeneratorKeras object can be passed.
            For the best speed the data_preparation method should be called before calling this function, if
            the data is in CVR ImageGeneratorKeras object.
        shuffle: If True, it shuffles the images at the end of each epoch
        random_state: The random state used in shuffling and oversampling
        over_sample: If True, it oversamples the smaller label
        return_image: If True, it outputs (image, image), If False it outputs (image, label)
        label: The name of the data generator
        image_col: The column in the df containing the 3D images
        y: The column in the df containing the labels
        random_rotation: If True it randomly rotates the atoms before generating the images
        **kwargs:
    """

    def __init__(self, data, batch_size=32, shuffle=False, random_state=0, over_sample=False, return_image=False,
                 label=None, image_col='image', y='y', image_params=None, verbose=True,
                 random_rotation=False, **kwargs):
        data = prepare_df(data)
        assert isinstance(data, pd.DataFrame), 'df must be a pandas DataFrame'
        # check the index is not duplicated
        assert data.index.is_unique, 'The index of df must be unique'
        self.index = pd.Series(dtype='int64')  # Valid samples index
        self.df = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.return_image = return_image
        self.label = label or ''
        self.image_col = image_col
        self.y = y
        self.random_rotation = random_rotation
        self.repeats = 1
        self.over_sample = over_sample
        self.training = False
        # self.verbose = verbose
        self.info = {}
        # self.image_params = image_params
        self.image_processor = ImagePreprocessor(data=self.df,
                                                 image_params=image_params,
                                                 verbose=verbose,
                                                 random_rotation=self.random_rotation,
                                                 )
        self.params = kwargs.get('generator_params', {})

        if kwargs.get('classification', False):
            self.params['classification'] = True
            self.df[self.y] = self.df[self.y].astype('category')

        self.set_index()

        if self.y not in self.df:
            self.df[self.y] = 1

        if over_sample:
            if verbose:
                print('Before oversampling: ', end='')
            self.stats()
            self.over_sampler()
            if verbose:
                print('After oversampling: ', end='')
            self.stats()

    def __len__(self):  # number of batches per epoch
        original_length = (len(self.index) + self.batch_size - 1) // self.batch_size
        # return original_length
        return original_length * self.repeats

    def __getitem__(self, idx):
        df = self.get_batch(idx)
        try:
            img = [i.get_image(normalization=True, random_rotation=self.random_rotation)
                   for i in df[self.image_col]]
        except Exception as e:
            print(f'\n{"*" * 5}Exception in loading idx={idx}: ', str(e), f'\n{"*" * 5}', flush=True)
            utg.save_var(locals(), 'tmp/exception_util_image_generator_keras__getitem__.pkl')
            raise e
        img = np.expand_dims(img, axis=0).squeeze(axis=0)
        if self.return_image:
            return img, img

        label = df[self.y].to_numpy().astype('float32').reshape((-1, 1))

        if 'label_like_image' in self.params:
            b_shape = list(img.shape)
            b_shape[-1] = 1
            bb = np.ones(b_shape)
            for i, b in enumerate(bb):
                b *= label[i]
            label = bb

        return img, label

    def get_batch(self, idx):
        # base_idx = idx * self.batch_size  # Without considering repeats

        # base_idx = (idx // self.repeats) * self.batch_size  # Considering repeats
        # (repeats batches and then moves on to the next batch

        base_idx = (idx % (self.__len__() // self.repeats)) * self.batch_size  # Considering repeats
        # First goes through all the batches and then repeats

        # df = self.df[base_idx: base_idx + self.batch_size]  # This selects rows without considering the index
        ind = self.index[base_idx: base_idx + self.batch_size]  # This selects rows considering the index
        df = self.df.loc[ind]
        return df

    # def get_index_of_all_batches(self):
    #     ind = [self.get_batch(i).index for i in range(len(self))]
    #     return pd.Series(ind).explode()
    #
    # def compute_average_over_repeats(self, y_pred):
    #     df = pd.DataFrame({'y_pred': y_pred.flatten()}, index=self.get_index_of_all_batches())
    #     # y_pred = df.group by(df.index).mean().loc[self.df.index].to_numpy()
    #     df['group_count'] = df.groupby(df.index).cumcount()
    #     df_pivot = df.pivot(columns='group_count', values='y_pred').loc[self.df.index]
    #     # df_pivot.columns = [f"y_pred_{i}" for i in df_pivot.columns]
    #     y_pred = df_pivot.mean(axis=1).to_numpy().reshape((-1, 1))
    #     return y_pred

    def set_index(self):
        self.index = self.df.index.to_series()
        if 'is_valid' in self.df:
            self.index = self.df[self.df['is_valid']].index.to_series()
        self.params['original_index'] = self.index.copy()

    def on_epoch_end(self):
        if self.shuffle and self.training:
            self.index = self.index.sample(frac=1, random_state=self.random_state)

    def stats(self):
        """
        Prints the number of samples in each class.
        """
        df = self.df.loc[self.index]
        # if self.verbose:
        print(f'Tot. samples={len(df):6,}')
        if df[self.y].dtype.name == 'category':
            # if self.verbose:
            print(df[self.y].value_counts())

    def get_labels(self):
        """
        Returns the labels of the valid indices.
        """
        return self.df.loc[self.index, self.y].to_numpy().reshape(-1, 1)

    def over_sampler(self):
        self.set_index()
        df = self.df.loc[self.index]  # Selecting the valid indices

        # Find the indices of the smaller and larger classes
        labels, counts = np.unique(df[self.y], return_counts=True)
        smaller_class_idx = np.argmin(counts)
        bigger_class_idx = np.argmax(counts)
        smaller_indices = df[df[self.y] == labels[smaller_class_idx]].index
        bigger_indices = df[df[self.y] == labels[bigger_class_idx]].index

        # Use pandas sample to over-sample smaller indices
        over_sample_indices = smaller_indices.to_series().sample(n=(len(bigger_indices) - len(smaller_indices)),
                                                                 replace=True, random_state=self.random_state)
        combined_indices = pd.concat([bigger_indices.to_series(), smaller_indices.to_series(), over_sample_indices])

        self.index = combined_indices.sample(frac=1, random_state=self.random_state)
        self.params['original_index'] = self.index

    def data_preparation(self, verbose=None):
        """
        This function is called before training and validation to speed up the process of image creation
        by pre-processing the point clouds over many cores. This is done only once.
        """
        # with utg.mute_print(enabled=not verbose):
        # self.prepare_point_clouds()
        self.image_processor.verbose = verbose or self.image_processor.verbose
        self.image_processor.prepare_point_clouds()
        self.set_index()

    def get_df(self):
        return self.df.loc[self.index]

    def temporary_repeat(self, n_repeats):
        """
        with generator.temporary_repeat(50) as repeating_generator:
            predictions = model.predict(repeating_generator)
            predictions = repeating_generator.compute_average_over_repeats(predictions)
        :param n_repeats:
        :return:
        """
        self.params['original_repeats'] = self.repeats
        self.repeats = n_repeats
        assert self.random_rotation or (self.repeats == 1), 'Random rotation must be enabled for using ensemble'
        return self

    def unwrap_repeat(self, yp: np.ndarray,
                      ensemble: int = None,
                      return_all_ensembles=False,
                      include_invalid_samples=False,
                      save_to_df=True
                      ):
        """
        After using temporary_repeat, this function can be used to unwrap the repeats.
        Temporary repeat is used for computing ensemble averaging predictions over different rotations.
        """
        n_samples = (self.df['is_valid'].sum() if 'is_valid' in self.df else len(self.df))
        if ensemble is None:
            ensemble = yp.shape[0] // n_samples
        assert ensemble * n_samples == yp.shape[0]

        yp = yp.reshape((ensemble, -1)).T

        df = pd.DataFrame(np.nan, index=self.df.index, columns=range(yp.shape[1]))
        df.loc[self.index] = yp
        df.insert(0, 'mean', df.mean(axis=1))

        if save_to_df:
            self.df['yp'] = df['mean']

        if return_all_ensembles:
            y_out = df
        else:
            y_out = df['mean']

        if not include_invalid_samples:
            y_out = y_out.loc[self.index]

        return y_out

    def save(self, filepath):
        """
        Save the current state of the generator to a file.
        Args:
            filepath: The path where to save the generator state.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a generator state from a file and return a new generator with this state.
        Args:
            filepath: The path from where to load the generator state.
        Returns:
            generator: The generator with the loaded state.
        """
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        generator = cls(data=pd.DataFrame())  # initialize with an empty DataFrame
        generator.__dict__.update(state)
        return generator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.repeats = self.params['original_repeats']
        self.params.pop('original_repeats', None)

    def __repr__(self):
        total_samples = f'{len(self.df):,}'
        valid_len = f'{sum(self.df["is_valid"]):,}' if "is_valid" in self.df else 'N/A'
        name = str(self.label) if self.label else ""

        return f'<ImageGenerator: {name}, {valid_len} / {total_samples}>'


class ImageGeneratorKeras(ImageGenerator, Sequence):
    def __init__(self, data, batch_size=32, shuffle=False, random_state=0, over_sample=False, return_image=False,
                 label=None, image_col='image', y='y', image_params=None, verbose=True,
                 random_rotation=False, **kwargs):
        super().__init__(data, batch_size, shuffle, random_state, over_sample, return_image, label, image_col, y,
                         image_params, verbose, random_rotation, **kwargs)


if __name__ == '__main__':
    pass
