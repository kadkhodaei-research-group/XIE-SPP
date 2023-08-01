from setuptools import setup, find_packages
import os
from pathlib import Path

params = {}
with open("xiespp/_params.py") as fp:
    exec(fp.read(), params)

install_requires = []
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().splitlines()
install_requires = [r for r in install_requires if not r.startswith('#')]


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
    version=params['__version__'],
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
    # code_supervisor='Sara Kadkhodaei',
    # code_supervisor_email='sarakad@uic.edu',
    description='XIE-SPP: Crystal Image Encoder for Synthesis & Property Prediction',
    keywords='synthesis synthesizability crystal formation energy',
    python_requires='>=3.7',
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'synthesizability = xiespp.main:main_synthesizability',
            'xiespp_synthesizability = xiespp.main:main_synthesizability',
            'xiespp_formation_energy = xiespp.main:main_formation_energy',
        ]
    }
)
