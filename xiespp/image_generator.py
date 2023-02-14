# from utility.util_crystal import *
import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
import os


# import crystals_tools

class ImageGeneratorDataFrame(Sequence):
    def __init__(self, df, shuffle=False, random_state=0, over_sample=False, return_image=False, label='Unlabeled',
                 target_col='img_path'):
        self.path = None
        if isinstance(df, str):
            self.path = df
            df = pd.read_csv(df)
        assert isinstance(df, pd.DataFrame)
        self.df = df.copy()
        self.shuffle = shuffle
        self.random_state = random_state
        self.return_image = return_image
        self.label = label
        self.target_col = target_col
        self.normalizer = image_normalizer_3

        if 'img_path' in self.df:
            if not os.path.exists(self.df['img_path'].iloc[0]):
                self.df['img_path'] = self.df['img_path'] + '.npy'
        if 'y' not in self.df:
            self.df['y'] = 1

        self.over_sample = over_sample
        if over_sample:
            print('Before oversampling: ', end='')
            self.stats()
            self.over_sampler()
            print('After oversampling: ', end='')
            self.stats()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # img = np.load(self.df['img_path'].iloc[idx])
        img = self.load_image(self.df[self.target_col].iloc[idx])
        # img = image_normalizer(img)
        # img = image_normalizer_2(img)
        img = self.normalizer(img)
        if self.return_image:
            return img, img

        label = self.df['y'].iloc[idx]
        label = np.array(label).astype('float32').reshape((-1, 1))
        label[label < 0] = 0

        return img, label

    def load_image(self, path):
        if not isinstance(path, str):
            import crystals_tools
            img = crystals_tools.cif2image(path, save_to=None, ok_error=False, skip_min_atomic_dist=False)
            return img
        if path[-4:] == '.cif':
            import crystals_tools
            img = crystals_tools.cif2image(path, save_to=None, ok_error=False, skip_min_atomic_dist=False)
            return img
        return np.load(path)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.random_state)

    def stats(self):
        print('Tot. samples={:6,}  [Pos.={:6,} - Neg.={:6,}({:.1f}%)]'.format(
            len(self.df), len(self.df[self.df['y'] == 1]), len(self.df[self.df['y'] == -1]),
            len(self.df[self.df['y'] == -1]) / len(self.df) * 100
        ))

    def get_labels(self):
        return self.df['y'].to_numpy()

    def over_sampler(self):
        random_state = self.random_state
        df = self.df
        df_n = df[df['y'] < 0]
        df_p = df[df['y'] > 0]
        if len(df_n) < len(df_p):
            df_smaller = df_n
            df_bigger = df_p
        else:
            df_smaller = df_p
            df_bigger = df_n
        # df_smaller = df_smaller.sample(n=len(df_bigger), replace=True, random_state=0)
        # df = pd.concat([df_smaller, df_bigger]).sample(frac=1, random_state=0).reset_index(drop=True)

        df_over = df_smaller.sample(n=len(df_bigger) - len(df_smaller), replace=True, random_state=random_state)
        df = pd.concat([df_smaller, df_bigger, df_over]).sample(frac=1, random_state=random_state)

        self.df = df


def image_normalizer(image, dtype='float32'):
    image = np.array(image, dtype=dtype)
    image[:, :, :, :, 0] = image[:, :, :, :, 0] / 119
    if np.max(image[:, :, :, :, 1]) <= 18:
        image[:, :, :, :, 1] = image[:, :, :, :, 1] / 19
    else:
        # raise EnvironmentError('There is something wrong with the the indexing')
        # To normalize Lanthanoids and Antinoids
        ind_norm = image[:, :, :, :, 1] <= 18
        image[ind_norm] = image[ind_norm] / 19
        image[~ind_norm] = 2. / 19 + (image[~ind_norm] - 18) / 15 * 1. / 19
    image[:, :, :, :, 2] = image[:, :, :, :, 2] / 8
    return image


def image_normalizer_2(image, dtype='float32'):
    image = np.array(image, dtype=dtype)
    image[:, :, :, :, 0] = image[:, :, :, :, 0] / 119
    if np.max(image[:, :, :, :, 1]) <= 18:
        image[:, :, :, :, 1] = image[:, :, :, :, 1] / 19
    else:
        # To normalize Lanthanoids and Antinoids
        image_slice = image[:, :, :, :, 1]
        image_slice[image_slice <= 18] = image_slice[image_slice <= 18] / 19
        image_slice[image_slice > 18] = 2. / 19 + (image_slice[image_slice > 18] - 18) / 15 * 1. / 19
    image[:, :, :, :, 2] = image[:, :, :, :, 2] / 8

    assert ((image < 1) & (image >= 0)).all()
    return image


def image_normalizer_3(image, dtype='float32'):
    image = np.array(image, dtype=dtype)
    image[:, :, :, :, 0] = image[:, :, :, :, 0] / 119

    # To normalize Lanthanoids and Antinoids
    image_slice = image[:, :, :, :, 1]
    image_slice[image_slice > 18] = 3.5
    image[:, :, :, :, 1] = image[:, :, :, :, 1] / 19

    image[:, :, :, :, 2] = image[:, :, :, :, 2] / 8

    assert ((image < 1) & (image >= 0)).all()
    return image



'''
####################
# image = np.arange(1, 3**3+1, dtype=int).reshape((3,3,3))
# print(image)
# print('-'*20)
# print('Z=1 :')
# print(image[:,: ,1])
# print('-'*20)
# print('Z=1 ,Y=1:')
# print(image[:,1 ,1])
# print('-'*20)
# print('Y=1 :')
# print(image[:,1 ,:])
# print('-'*20)
# print('X=1 :')
# print(image[1,: ,:])
# print('-'*20)
# print('X=1,Y=1 :')
# print(image[1,1 ,:])
# print('-'*20)

# print('image:')
# print(image)
# print('-'*20)
# a = image[:,0 ,:]
# print(a)
# # a[a>10] /= 2 # Works for float
# a[a>10] = a[a>10]/2
# print(image)


####################

# image = np.arange(1, 3**3+1, dtype=int).reshape((3,3,3))
# print(image[:,:,0])
# print('-'*20)
# image[:,:,0] = image[:,:,0] / 10
# print(image)

###################

# Wrong 
# image = np.arange(1, 3**3+1, dtype=int).reshape((3,3,3))
# print(image)
# print('-'*20)
# print(image[:,0 ,:])
# print('-'*20)
# ind_norm = image[:,0 ,:] > 10
# image[ind_norm] = 0
# print(image)

###################
# Right
# image = np.arange(1, 3**3+1, dtype=int).reshape((3,3,3))
# print(image)
# print('-'*20)
# print(image[:,0 ,:])
# print('-'*20)
# image[:,0 ,:] = image[:,0 ,:] /2
# # ind_norm = image[:,0 ,:]
# # ind_norm[ind_norm>10] = 0
# print(image)
'''

# def prepare_generators()
