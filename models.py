from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation

from keras.models import Input, Model
from keras.layers import Conv3D, Conv2D, MaxPooling2D
from keras.layers import MaxPooling3D, UpSampling3D
from utility.util_tf import tf_shut_up
from keras import regularizers
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Conv3D, Flatten, Dense, MaxPooling3D, Dropout
from keras.regularizers import l2
import os


class CAE:
    def __init__(self,
                 input_shape,
                 channels=None,
                 kernel_size=(3, 3, 3),
                 activation=None,
                 pool_size=None,
                 loss='binary_crossentropy',
                 optimizer='rmsprop',
                 metrics=None,
                 verbose=True,
                 dropout_rate=None,
                 kernel_regularizer=None,
                 ):
        if channels is None:
            channels = [32, 32, 64]
        if activation is None:
            activation = ['relu', 'sigmoid']
        if pool_size is None:
            pool_size = 2
        if isinstance(pool_size, int):
            pool_size = [pool_size] * len(channels)
        if isinstance(pool_size, list):
            pool_size = [(i, i, i) for i in pool_size]

        self.input_shape = input_shape
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool_size = pool_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer

    def generate(self):
        if self.verbose:
            print('\nMODEL 5 adjustable params:')
            print('\n'.join([str(i) + ': ' + str(j) for i, j in self.__dict__.items()]) + '\n')

        channels = self.channels.copy()
        pool_size = self.pool_size.copy()

        mod = Sequential()
        mod.add(layers.InputLayer(input_shape=self.input_shape))

        for e in ['encoded', 'decoded']:
            if e == 'decoded':
                channels.reverse()
                channels.append(self.input_shape[-1])
                pool_size.reverse()
            for i in range(len(channels)):
                name = None
                decoding_starting_point = None
                activation = self.activation[0]
                if i == len(channels) - 1:
                    name = e
                    if e == 'decoded':
                        activation = self.activation[1]
                if (i == 0) & (e == 'decoded'):
                    decoding_starting_point = 'decoding'

                mod.add(layers.Conv3D(filters=channels[i], kernel_size=self.kernel_size, padding='same',
                                      name=decoding_starting_point, kernel_regularizer=self.kernel_regularizer))
                mod.add(layers.Activation(activation))
                if e == 'encoded':
                    mod.add(layers.MaxPooling3D(pool_size=pool_size[i], name=name))
                if e == 'decoded' and (not i == len(channels) - 1):
                    mod.add(layers.UpSampling3D(pool_size[i]))
                if self.dropout_rate is not None:
                    mod.add(Dropout(self.dropout_rate))

        mod.compile(loss=self.loss,
                    optimizer=self.optimizer,
                    metrics=self.metrics
                    )
        if self.verbose > 1:
            mod.summary()
        return mod

    def generate_and_initialize_weights(self, weight_file):
        cae_1 = self.generate()
        if not os.path.exists(weight_file):
            raise FileNotFoundError('weight file was not found!')
        cae_1.load_weights(weight_file)
        return cae_1

    def generate_encoder(self, weigh_file):
        cae_1 = self.generate_and_initialize_weights(weigh_file)
        for i in range(len(cae_1.layers)):
            if ('encode' in cae_1.layers[-1].name) or ('pooling' in cae_1.layers[-1].name):
                break
            cae_1.pop()
        cae_1.add(layers.Flatten())
        return cae_1


if __name__ == '__main__':
    print('The End!')
