# from .generator import ImageGeneratorKeras
from .generator import ImageGeneratorKeras


def predict(model,
            generator,
            params=None,
            ensemble=None,
            device=None,
            verbose=1,
            workers=32,
            include_invalid_samples=False,
            return_all_ensembles=False,
            **kwargs):
    """
    Predicts the output of a model using a rotational ensemble averaging.
    :param model: Keras model
    :param generator: CVR.ImageGeneratorKeras object.
    For the best speed the data_preparation method should be called before calling this function.
    :param params: Dictionary of parameters. The default value is {}.
    :param ensemble: The default value is 50. Increasing the ensemble size will increase the precision but will also
    increase the prediction time.
    :param device: To use GPU pass '/device:GPU:0' or '/device:GPU:1' etc. To use CPU pass '/device:CPU:0'
    CPU is the default value.
    :param verbose: Keras verbose parameter
    :param workers: Keras workers parameter
    :param include_invalid_samples: Some samples may not meet the requirements of the CVR images.
    If true NaN values will be returned for those samples. If false those samples will be ignored.
    :param return_all_ensembles: All the predictions of the ensemble will be returned.
    :param kwargs: The rest of Keras predict parameters
    :return: Predictions. Data Frame will be returned if return_all_ensembles is True.
    Otherwise, a numpy array will be returned.
    """
    if params is None:
        params = {}
    if device is None:
        device = params.get('device', '/device:CPU:0')
    if ensemble is None:
        ensemble = params.get('ensemble', 1)

    import tensorflow as tf
    with generator.temporary_repeat(ensemble) as gen_ensemble:
        with tf.device(device):
            yp = model.predict(
                gen_ensemble,
                verbose=verbose,
                workers=workers,
                steps=None,  # To prevent user from passing steps because it doesn't support ensemble
                **kwargs,
            )

        yp = generator.unwrap_repeat(yp=yp,
                                     return_all_ensembles=return_all_ensembles,
                                     include_invalid_samples=include_invalid_samples
                                     )
    return yp


def prepare_data(
        data_input,
        image_params=None,
        input_format=None,
        verbose=True,
) -> ImageGeneratorKeras:
    """
    Prepares data for prediction. Makes sure that the data is in CVR ImageGeneratorKeras object.
    :param data_input: List of files - ASE atomic objects - ThreeDImages - PyMatGen Structure objects
    :param image_params: Image settings
    # :param random_rotation: Has to be on to use rotational ensemble
    :param input_format: The format of the files in data_input
    :param verbose: True to print progress
    :return: CRV. Image Generator object for Keras models
    """
    if input_format is not None:
        image_params = image_params.copy()
        image_params['format'] = input_format

    if not isinstance(data_input, ImageGeneratorKeras):
        data_input = ImageGeneratorKeras(
            data_input,
            image_params=image_params,
            random_rotation=image_params.get('random_rotation', True),
            verbose=verbose,
        )
        data_input.data_preparation()
    return data_input


class BasePredictor:
    """
    Base Predictor
    """
    def __init__(self, device=None, ensemble=None, batch_size=None, image_params=None):
        if ensemble is None:
            ensemble = 50
        if batch_size is None:
            batch_size = 32
        self.model = self.get_model(load_weights=True)
        self.device = device or '/device:CPU:0'  # default to CPU if no GPU device specified
        self.ensemble = ensemble
        self.batch_size = batch_size
        self.image_params = image_params

    def get_model(self, load_weights):
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self,
                generator,
                include_invalid_samples=True,
                return_all_ensembles=False,
                verbose=1,
                input_format=None,
                **kwargs):
        """
        Makes predictions using the model and the CVR ImageGeneratorKeras object.
        :param generator: CVR.ImageGeneratorKeras object.
        For the best speed the data_preparation method should be called before calling this function.
        :param include_invalid_samples: Some samples may not meet the requirements of the CVR images.
        If true NaN values will be returned for those samples. If false those samples will be ignored.
        :param return_all_ensembles: All the predictions of the ensemble will be returned.
        :param verbose: Keras verbose parameter
        :param input_format: The format of the files in data_input (Use ASE format names)
        :param kwargs: The rest of Keras predict parameters
        """
        generator = prepare_data(
            generator,
            image_params=self.image_params,
            verbose=verbose > 0,
            input_format=input_format
        )
        generator.batch_size = self.batch_size
        return predict(
            model=self.model,
            generator=generator,
            ensemble=self.ensemble,
            device=self.device,
            include_invalid_samples=include_invalid_samples,
            verbose=verbose,
            return_all_ensembles=return_all_ensembles,
            **kwargs
        )

    def __repr__(self):
        return f'<BasePredictor: device={self.device}, ensemble={self.ensemble}>'





