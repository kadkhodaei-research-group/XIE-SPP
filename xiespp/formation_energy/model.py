# from .. import CVR


def get_model(load_weights=True):
    """
    Returns the model used for the formation energy prediction.
    :param load_weights: Whether to load the weights of the model.
    :return: Keras model
    """
    from keras import layers
    from keras.models import Model

    n_bins = 32
    n_channels = 3

    input_shape = tuple([n_bins] * 3 + [n_channels])
    inputs = layers.Input(shape=input_shape)
    lc = inputs  # Should not be removed

    lc = convolutional_block(lc, num_filters=32, kernel_size=3, activation='relu')

    lc = residual_block(lc, n_conv_blocks=2, n_filters=32, kernel_size=3, activation='relu', merge_mode='add')
    lc = residual_block(lc, n_conv_blocks=2, n_filters=32, kernel_size=3, activation='relu', merge_mode='concat')

    lc = layers.AveragePooling3D()(lc)

    lc = residual_block(lc, n_conv_blocks=2, n_filters=64, kernel_size=3, activation='relu', merge_mode='add')
    lc = residual_block(lc, n_conv_blocks=2, n_filters=64, kernel_size=3, activation='relu', merge_mode='add')
    lc = residual_block(lc, n_conv_blocks=2, n_filters=64, kernel_size=3, activation='relu', merge_mode='concat')

    lc = layers.AveragePooling3D()(lc)

    lc = residual_block(lc, n_conv_blocks=2, n_filters=128, kernel_size=3, activation='relu', merge_mode='add')
    lc = residual_block(lc, n_conv_blocks=2, n_filters=128, kernel_size=3, activation='relu', merge_mode='add')

    lc = layers.AveragePooling3D()(lc)
    lc = layers.Flatten()(lc)

    lc = layers.Dense(16)(lc)
    lc = layers.Dense(16)(lc)
    lc = layers.Dense(1)(lc)

    outputs = lc
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mean_absolute_error']
                  )

    if load_weights:
        from pathlib import Path

        # Method 1
        model_path = Path(__file__).parent / 'model/weights-picked.h5'

        # Method 2
        # import pkg_resources
        # model_path = pkg_resources.resource_filename('xiespp.formation_energy', 'model/weights-picked.h5')

        # Method 3
        if not model_path.exists():
            import importlib_resources
            model_path = importlib_resources.files('xiespp.formation_energy') / 'model/weights-picked.h5'

        assert Path(model_path.__str__()).exists(), f'Weights file {model_path} does not exist'
        model.load_weights(model_path)

    # model.summary()
    return model


def convolutional_block(input_stream, num_filters: int, stride: int = 1, kernel_size: int = 3,
                        activation='relu', bn: bool = False):
    """
    Parameters
    ----------
    input_stream : Tensor layer
        Input tensor from previous layer
    num_filters : int
        Conv.3d number of filters
    stride : int by default 1
        Stride square dimension
    kernel_size : int by default 3
        COnv2D square kernel dimensions
    activation: str by default 'relu'
        Activation function to used
    bn: bool by default True
        To use BatchNormalization or not
    """
    from keras import layers
    conv_layer = layers.Conv3D(num_filters,
                               kernel_size=kernel_size,
                               strides=stride,
                               padding='same',
                               )
    tensor_x = conv_layer(input_stream)
    if bn:
        tensor_x = layers.BatchNormalization()(tensor_x)
    if activation is not None:
        tensor_x = layers.Activation(activation)(tensor_x)
        # lc = Dropout(0.2)(lc)
    return tensor_x


def residual_block(input_stream, n_conv_blocks=2, n_filters=16, kernel_size=3, activation=None, merge_mode='off'):
    """
    Residual block for 3D CNNs.
        :param input_stream: Input stream to the residual block.
        :param n_conv_blocks: Number of convolutional blocks in the residual block.
        :param n_filters: Number of filters in the convolutional layers.
        :param kernel_size: Kernel size of the convolutional layers.
        :param activation: Activation function of the convolutional layers.
        :param merge_mode: Merge mode of the residual block.
                auto: Merge mode is automatically determined based on the number of layers in the block.
                add: Add the input and output.
                concat: Concatenate the input and output.
                off(default): Do not merge the input and output.
    """
    from keras import layers
    x = input_stream

    for i in range(n_conv_blocks):
        x = convolutional_block(x, n_filters, kernel_size=kernel_size, activation=activation)

    # Merging the input stream with the output stream of the residual block
    if merge_mode == 'auto':
        if input_stream.shape[-1] == x.shape[-1]:
            merge_mode = 'add'
        else:
            merge_mode = 'concat'
    if merge_mode == 'add':
        x = layers.Add()([x, input_stream])
    if merge_mode == 'concat':
        x = layers.Concatenate()([x, input_stream])
    # if merge_mode == 'conv':
    #     conv = layers.Conv3D(n_filter, kernel_size,
    #                          padding=padding,
    #                          strides=2,
    #                          kernel_initializer=kernel_initializer,
    #                          )
    #     x_side = conv(input_stream)
    #     x = layers.Add()([x, x_side])
    if merge_mode == 'off':
        pass

    return x
