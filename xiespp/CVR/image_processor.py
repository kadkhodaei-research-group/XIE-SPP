from . import util_cvr as utg
from . import crystal_image_tools
from .util_crystal import crystal_parser
import pandas as pd
import numpy as np


class ImagePreprocessor:
    """
    A class for preparing 3D images from CIF files.
    This class helps ImageGeneratorKeras to pre-process the images.
    """

    def __init__(self,
                 data,
                 image_params: dict,
                 random_rotation=True,
                 verbose=True):
        self.df = prepare_df(data)
        self.image_params = image_params
        self.verbose = verbose
        self.random_rotation = random_rotation

    def read_files(self, input_col='file', output_col='atoms'):
        if self.verbose:
            print('Reading CIF files...', flush=True)
        df = self.df
        df[output_col] = [crystal_parser(filepath=f, format=self.image_params.get('format', None))
                          for f in utg.pbar(df[input_col], verbose=self.verbose)]

    def image_preparation(self, input_col='atoms', check_requirements=True):
        # df = self.df_all
        df = self.df
        if input_col not in df.columns:
            self.read_files()

        if self.verbose:
            print('Preparing image objects...', flush=True)
        assert self.image_params is not None, 'image_params must be provided to prepare images'
        if 'box' not in self.image_params:
            assert 'box_size' in self.image_params and 'n_bins' in self.image_params, \
                'box or box_size and n_bins must be provided'
            self.image_params['box'] = crystal_image_tools.BoxImage(
                box_size=self.image_params['box_size'],
                n_bins=self.image_params['n_bins']
            )
        assert 'box' in self.image_params, 'box must be provided'
        assert 'channels' in self.image_params, 'channels must be provided'
        assert 'filling' in self.image_params, 'filling type must be provided'

        img = [crystal_image_tools.ThreeDImage(atoms=r[input_col], **self.image_params) for _, r in df.iterrows()]
        df.loc[df.index, 'image'] = img

        if check_requirements:
            self.check_image_requirements()

    def check_image_requirements(self):
        # df = self.df_all
        df = self.df

        if self.verbose:
            print('Checking image requirements...', flush=True)
        df.loc[df.index, 'is_valid'] = utg.parallel_apply(
            df.loc[:, 'image'], lambda x: x.check_requirements(),
            progres_bar=self.verbose,
        )
        df['is_valid'] = df['is_valid'].astype('bool')
        if self.verbose:
            print('Valid images: {:7,} / {:7,}'.format(len(df[df['is_valid']]), len(df)), flush=True)
        # self.df = df[df['is_valid']]
        # self.set_index()

    def prepare_point_clouds(self, check_requirements=True):
        if 'image' not in self.df.columns:
            self.image_preparation(check_requirements=check_requirements)
        # df = self.df.loc[self.index]
        df = self.df[self.df['is_valid']]

        if check_requirements:
            assert all(df['is_valid']), 'Some images are not valid'

        if self.verbose:
            print('Preparing point clouds...', flush=True)
        tmp = utg.parallel_apply(
            df.loc[:, 'image'],
            lambda x, random_rotation: x.get_point_cloud(random_rotation=random_rotation),
            random_rotation=self.random_rotation,
            progres_bar=self.verbose,
        )
        for n, i in enumerate(df.index):
            df.loc[i, 'image'].set_point_cloud(tmp[n])


def prepare_df(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, (str, utg.Path)):
        df = pd.read_csv(df)
    if isinstance(df, (list, tuple, np.ndarray, pd.Series)):
        column_name = pick_the_column_name(df[0])  # file or atoms
        df = pd.DataFrame({column_name: df})
    return df


def pick_the_column_name(obj):
    column_name = 'file'
    module_name = obj.__class__.__module__
    packages = ['pydantic', 'pymatgen', 'ase']
    if any(pkg in module_name for pkg in packages):
        column_name = 'atoms'
    return column_name
