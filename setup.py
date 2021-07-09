from setuptools import setup, find_packages
import os, glob, shutil, distutils


install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().splitlines()

# for f in glob.glob('*.py'):
#     shutil.copyfile(f, 'synthesizability/' + f)
# distutils.dir_util.copy_tree('utility', 'synthesizability')
# distutils.dir_util.copy_tree('finalized_results', 'synthesizability')
file1 = open("synthesizability/synthesizability.py", "a")
file1.write(f'\nrep_dir = "{os.path.dirname(os.path.realpath(__file__))}"\n')
file1.close()

setup(
    name='synthesizability',
    version='1.0.0',
    packages=find_packages(),
    # packages=['synthesizability'],
    # package_dir={'synthesizability': './'},
    url='https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor',
    license='CC BY-ND 4.0',
    author='Ali Davari, Sara Kadkhodaei',
    author_email='adavar2@uic.edu, sarakad@uic.edu',
    description='Neural Network Crystal Synthesizability Predictor (NNCSP)',
    keywords='synthesis synthesizability crystal',
    # long_description=open('README.md').read(),
    python_requires='>=3.7',
    install_requires=install_requires,
)
