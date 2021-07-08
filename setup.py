from setuptools import setup, find_packages
import os


install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().splitlines()

setup(
    name='synthesizability_predictor',
    version='1.0.0',
    packages=['synthesizability'],
    package_dir={'synthesizability': './'},
    url='https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor',
    license='CC BY-ND 4.0',
    author='Ali Davari, Sara Kadkhodaei',
    author_email='adavar2@uic.edu, sarakad@uic.edu',
    description='Neural Network Crystal Synthesizability Predictor (NNCSP)',
    keywords='synthesis synthesizability crystal',
    # long_description=open('README.md').read(),
    python_requires='>=3.7.3',
    install_requires=install_requires,

)
