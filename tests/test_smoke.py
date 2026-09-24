"""
Smoke tests: exact recreation of the CVR images and reproduction of the model predictions.

    uv run pytest tests      (or: pytest tests)

The CVR images (no rotation) are deterministic and are compared by their SHA-256 hash.
Synthesizability v1 has no random rotations and is compared almost exactly.
Synthesizability v2 and formation energy average randomly rotated images, so the predictions are compared
with a tolerance of ~3x the spread observed between repeated runs.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from xiespp import CVR, formation_energy, get_test_samples, synthesizability, synthesizability_1

# SHA-256 of the float32 image arrays, created with the default image parameters of each model.
IMAGE_HASHES = {
    ('GaN_12.cif', 'synthesizability'): 'bf9042508c9c5a706bc18c74ae7db317fca59540de44f89ee6658f4cb5482646',
    ('GaN_12.cif', 'formation_energy'): '77c7647c84b8faeb31a0ad3a0ec79d9ff146c515e165ebd9b741f8bfee2f7c0c',
    ('GaN_9.cif', 'synthesizability'): '3a229a7ea604336ce183501f7f60c383eaf5147263e17264097a6715a6562003',
    ('GaN_9.cif', 'formation_energy'): '72d867259ccb1e00d52ff55d317759acc8606794d6b3207f3f2322e01ec5ef49',
    ('CSi_12.cif', 'synthesizability'): '3125fff96e56cb7e3e750d7c8562126c53243a1440c22c2c5828bf38aa431ad1',
    ('CSi_12.cif', 'formation_energy'): 'a1585d1fd3ccccf03a5867f8bed4b5b24d6a93c0849f68d3c7dc5d49588494fe',
    ('CSi_84.cif', 'synthesizability'): 'fb7bd74d4ec728e99ffdc0100a75998d346031766cc2cd3c21e2ca6c3b5b49dc',
    ('CSi_84.cif', 'formation_energy'): '881ca4bafe0661dce9e46a9390dba25d16817c8127536116c711b4cd580019c8',
}
IMAGE_PARAMS = {
    'synthesizability': synthesizability.IMAGE_PARAMS,
    'formation_energy': formation_energy.DEFAULT_IMAGE_PARAMS,
}

# Reference predictions (inputs sorted by file name)
SYN_V1 = {  # GaN_12, GaN_9
    'cae-mlp': [0.96435136, 0.00200528],
    'cnn': [0.00150415, 0.00446968],
}
SYN_V2_CSI = [0.2181, 0.9939]  # CSi_12, CSi_84 (mean of 6 runs, max deviation 0.033)
SYN_V2_TOL = 0.1
FE_MOS2 = [-1.0542, -1.0679, -0.9747, -1.0741, -0.4087, -0.9081, -0.8437, -0.7886, -0.5243]  # eV/atom
FE_TOL = 0.07  # max deviation between runs: 0.022 eV/atom


def samples(name):
    return sorted(get_test_samples(name))


def test_samples_available():
    assert [Path(f).name for f in samples('GaN')] == ['GaN_12.cif', 'GaN_9.cif']
    assert [Path(f).name for f in samples('CSi')] == ['CSi_12.cif', 'CSi_84.cif']
    assert len(samples('MoS2')) == 9


@pytest.mark.parametrize('file_name, preset', sorted(IMAGE_HASHES))
def test_cvr_image_exact(file_name, preset):
    f = next(f for f in samples(file_name.split('_')[0]) if Path(f).name == file_name)
    params = IMAGE_PARAMS[preset]
    image = CVR.ThreeDImage(
        atoms=CVR.crystal_parser(filepath=f),
        box=CVR.BoxImage(box_size=params['box_size'], n_bins=params['n_bins']),
        channels=params['channels'],
        filling=params['filling'],
    ).get_image()
    assert image.dtype == np.float32
    assert image.shape == (params['n_bins'],) * 3 + (len(params['channels']),)
    assert hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() == IMAGE_HASHES[file_name, preset]


@pytest.mark.parametrize('classifier', sorted(SYN_V1))
def test_synthesizability_v1(classifier):
    yp = synthesizability_1.synthesizability_predictor(samples('GaN'), classifier=classifier, verbose=0)
    np.testing.assert_allclose(yp, SYN_V1[classifier], atol=1e-5)


def test_synthesizability_v2():
    yp = synthesizability.SynthesizabilityPredictor().predict(samples('CSi'), verbose=0)
    np.testing.assert_allclose(yp.values, SYN_V2_CSI, atol=SYN_V2_TOL)


def test_formation_energy():
    yp = formation_energy.FormationEnergyPredictor().predict(samples('MoS2'), verbose=0)
    np.testing.assert_allclose(yp.values, FE_MOS2, atol=FE_TOL)


def test_cli(tmp_path):
    from xiespp.main import main
    import pandas as pd

    out = tmp_path / 'out.csv'
    main(['synthesizability', '--test-samples', '--model', 'v1-cnn', '-o', str(out)])
    df = pd.read_csv(out)
    assert [Path(f).name for f in df['file']] == ['GaN_12.cif', 'GaN_9.cif']
    np.testing.assert_allclose(df['synthesizability'], SYN_V1['cnn'], atol=1e-5)
