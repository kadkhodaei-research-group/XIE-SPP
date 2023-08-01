CONDA_ENV_NAME=syn

set -x

# conda create -n $CONDA_ENV_NAME python=3.7 -y
# source ~/.bash_profile
# conda activate $CONDA_ENV_NAME
echo -e "\033[32m creating conda environment $CONDA_ENV_NAME \033[0m"; sleep 5
conda create -n $CONDA_ENV_NAME python=3.8 -y
source activate $(conda info --base)/envs/$CONDA_ENV_NAME
echo -e "\033[32m Active conda environment: $CONDA_DEFAULT_ENV \033[0m"; sleep 5

# source prepare_jupyter.sh

if [[ $OSTYPE == 'darwin'* ]]; then
  echo 'macOS detected'
  echo 'miniforge3 is required'
  # Pause for 5 seconds
  sleep 5
  # conda install -c conda-forge keras=2.4.3 tensorflow=2.4.1 -y
  # conda install -c conda-forge keras=2.4.3 tensorflow=2.4.1 -y
  conda install -c apple tensorflow-deps -y
  pip install tensorflow-macos
  pip install tensorflow-metal
else
  echo 'Linux detected'
  conda install -c anaconda keras-gpu=2.4.3 tensorflow=2.4.1 -y
fi

# source crystall_essentials.conda.sh
#conda install seaborn tqdm scipy matplotlib bokeh numba scikit-learn=0.24.2 -y
# conda install scikit-learn=0.24.2 -y
# pip install ase scikit-learn ipynbname
#pip install ase ipynbname

# git clone https://github.com/kadkhodaei-research-group/XIE-SPP.git
# cd XIE-SPP

python setup.py install

# conda clean --all -y

# Usage:
# source synthesizability.conda.sh
# conda activate $CONDA_ENV_NAME
# which xiespp_synthesizability  # Executable path
# xiespp_synthesizability --help
# xiespp_synthesizability --test -v