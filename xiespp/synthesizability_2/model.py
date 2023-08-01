from keras.models import Sequential  # , load_model
from keras import layers
# from keras.optimizers import Adam
# import keras.backend as K


def get_cnn_model(params, summary=True, weights=None):
    image_params = params.get('image_params', params)
    mlp_nodes_p_layer = params['mlp_nodes_p_layer']
    input_shape = (
        image_params['n_bins'],
        image_params['n_bins'],
        image_params['n_bins'],
        len(image_params['channels'])
    )
    opti = 'adam'

    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    # Encoder:

    model.add(layers.Conv3D(filters=32, kernel_size=3, padding='same', name=None))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(4))

    model.add(layers.Conv3D(filters=32, kernel_size=3, padding='same', name=None))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(4))

    model.add(layers.Conv3D(filters=32, kernel_size=3, padding='same', name=None))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(4, name='encoded'))

    model.add(layers.Flatten())
    model.add(layers.Dense(mlp_nodes_p_layer))
    model.add(layers.Dense(mlp_nodes_p_layer))
    model.add(layers.Dense(mlp_nodes_p_layer))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=opti,
        metrics=['accuracy'],
    )

    if summary:
        model.summary()

    if weights:
        model.load_weights(weights)
    return model
