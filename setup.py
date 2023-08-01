from setuptools import setup, find_packages
import os
from pathlib import Path
from xiespp.params import __version__ as version


install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().splitlines()


# data_dir = os.path.join('finalized_results')
# datafiles = [(d, [os.path.join(d, f) for f in files])
#              for d, folders, files in os.walk(data_dir)]
def package_files(directory, pattern='**/*'):
    paths = []
    for filepath in Path(directory).glob(pattern):
        if filepath.is_file():
            paths.append(str(filepath.relative_to(directory)))
    return paths


setup(
    name='xiespp',
    # data_files=datafiles,
    version=version,
    packages=find_packages(),
    package_data={
        'xiespp.formation_energy': ['model/*'],
        'xiespp.CVR': ['periodic_table.csv'],
        'xiespp.synthesizability_1':
            package_files('xiespp/synthesizability_1', pattern='data/**/*.cif')
            + package_files('xiespp/synthesizability_1', pattern='models/**/*.pkl')
            + package_files('xiespp/synthesizability_1', pattern='models/**/*.h5'),
        'xiespp.synthesizability_2': ['model/*'],
    },
    url='https://github.com/kadkhodaei-research-group/XIE-SPP',
    license='CC BY-ND 4.0',
    author='Ali Davariashtiyani',
    author_email='adavar2@uic.edu, ',
    code_supervisor='Sara Kadkhodaei',
    code_supervisor_email='sarakad@uic.edu',
    description='XIE-SPP: Crystal Image Encoder for Synthesis & Property Prediction',
    keywords='synthesis synthesizability crystal formation energy',
    # long_description=open('README.md').read(),
    python_requires='>=3.7',
    # install_requires=install_requires,  # TODO: accept different versions of tensorflow
    # https://stackoverflow.com/questions/49222824/make-an-either-or-distinction-for-install-requires-in-setup-py
    entry_points={
        'console_scripts': [
            'synthesizability = xiespp.main:main_synthesizability',
            'xiespp_synthesizability = xiespp.main:main_synthesizability',
            'xiespp_formation_energy = xiespp.main:main_formation_energy',
        ]
    }
)
