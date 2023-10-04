# formation_energy/predictor.py
from . import model as model_module
# from . import CVR
import CVR
import functools


DEFAULT_RANDOM_ROTATION = True
DEFAULT_IMAGE_PARAMS = dict(
    box_size=70 // 4,
    n_bins=128 // 4,
    channels=['atomic_number', 'group', 'period'],
    filling='fill-cut',
    random_rotation=DEFAULT_RANDOM_ROTATION,
)
DEFAULT_BOX = CVR.BoxImage(box_size=DEFAULT_IMAGE_PARAMS['box_size'], n_bins=DEFAULT_IMAGE_PARAMS['n_bins'])

prepare_data = functools.partial(
    CVR.keras_tools.prepare_data,
    image_params=DEFAULT_IMAGE_PARAMS,
)

class FormationEnergyPredictor(CVR.keras_tools.BasePredictor):
    """
    Predicts formation energy of a crystal structure from its atomic structure
    using rotational ensemble of 50 (default value) CVR images.
    The tensorflow model is loaded from the default location.
    """
    def __init__(self, device=None, ensemble=None, batch_size=None, image_params=None):
        if batch_size is None:
            batch_size = 32
        if ensemble is None:
            ensemble = 50
        if image_params is None:
            image_params = DEFAULT_IMAGE_PARAMS
        super().__init__(device=device, ensemble=ensemble, batch_size=batch_size, image_params=image_params)

    def get_model(self, load_weights):
        return model_module.get_model(load_weights)

    def __repr__(self):
        return f'<FormationEnergyPredictor: device={self.device}, ensemble={self.ensemble}>'
