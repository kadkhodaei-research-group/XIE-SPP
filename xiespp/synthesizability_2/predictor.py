from . import model as model_module
from .params import IMAGE_PARAMS
from pathlib import Path
from .. import CVR


class SynthesizabilityPredictor(CVR.keras_tools.BasePredictor):
    """
    Predicts synthesizability likelihood of crystal structures from its atomic structure
    using rotational ensemble of 100 (default value) CVR images.
    The tensorflow model is loaded from the default location.
    """
    def __init__(self, device=None, ensemble=None, batch_size=None, image_params=None):
        if batch_size is None:
            batch_size = 8
        if ensemble is None:
            ensemble = 100
        if image_params is None:
            image_params = IMAGE_PARAMS
        super().__init__(device=device, ensemble=ensemble, batch_size=batch_size, image_params=image_params)

    def get_model(self, load_weights):
        model_params = {
            'image_params': IMAGE_PARAMS,
            'mlp_nodes_p_layer': 13,
        }
        weights_path = None
        if load_weights:
            weights_path = Path(__file__).parent / 'model/weights0500.h5'
        return model_module.get_cnn_model(params=model_params, weights=weights_path, summary=False)

    def __repr__(self):
        return f'<SynthesizabilityPredictor: device={self.device}, ensemble={self.ensemble}>'
