from setuptools import setup, find_packages
import os
import xiespp

# import glob, shutil, distutils

install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().splitlines()


data_dir = os.path.join('finalized_results')
datafiles = [(d, [os.path.join(d, f) for f in files])
             for d, folders, files in os.walk(data_dir)]

setup(
    name='xiespp',
    data_files=datafiles,
    version=xiespp.__version__,
    packages=find_packages(),
    url='https://github.com/kadkhodaei-research-group/XIE-SPP',
    license='CC BY-ND 4.0',
    author='Ali Davariashtiyani',
    author_email='adavar2@uic.edu, ',
    code_supervisor='Sara Kadkhodaei',
    code_supervisor_email='sarakad@uic.edu',
    description='XIE-SPP: Crystal Image Encoder for Synthesis & Property Prediction',
    keywords='synthesis synthesizability crystal',
    # long_description=open('README.md').read(),
    python_requires='>=3.7',
    install_requires=install_requires,  # TODO: accept different versions of tensorflow
    # https://stackoverflow.com/questions/49222824/make-an-either-or-distinction-for-install-requires-in-setup-py
    entry_points={
        'console_scripts': [
            'synthesizability = xiespp.synthesizability:main'
        ]
    }
)
