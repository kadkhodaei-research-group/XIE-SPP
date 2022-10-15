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

# file1 = open("synthesizability/synthesizability.py", "a")
# file1.write(f'\nrep_dir = "{os.path.dirname(os.path.realpath(__file__))}"\n')
# file1.close()

setup(
    name='xiespp',
    include_package_data=True,
    version='1.0.6',
    packages=find_packages(),
    # packages=['synthesizability'],
    # package_dir={'synthesizability': './'},
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
    install_requires=install_requires,
)
