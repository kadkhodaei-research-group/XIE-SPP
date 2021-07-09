# Neural Network Crystal Synthesizability Predictor (NNCSP)

![https://creativecommons.org/licenses/by-nd/4.0/](https://i.creativecommons.org/l/by-nd/4.0/88x31.png)
[LICENSE](https://creativecommons.org/licenses/by-nd/4.0/)
## 1. Abstract

Predicting the synthesizability of hypothetical crystals has proven to be challenging due to the wide range of parameters that govern crystalline materials synthesis. In this work, we convert the atomic structures of known synthesized or observed crystals in databases into three-dimensional pixel-wise images color-coded by chemical species and electronegativity and use them for training a deep-network convolutional. We extract the latent features of synthesizability hidden in structural and chemical arrangements of crystalline materials embedded in the auto-encoder. The accurate classification of materials into synthesizable crystals vs. crystal anomalies based on these features across a broad range of crystal structure types and chemical compositions confirms the validity of our model.

## 2. Installation
Tested on Ubuntu 20.04

Navigating to the desired directory in the terminal:
~~~sh
cd [/navigate/to/directory/]
~~~

Cloning the repository and entering the directory:
~~~sh
git clone https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor.git
cd NN-crystal-synthesizability-predictor/
~~~

We highly encourage you to use a [conda envirenment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you don't have conda installed on your machine, you can skip this step:
~~~sh
conda create -n synthesizability python=3.7.3 -y
conda activate synthesizability
~~~

Installing TensorFlow:
~~~sh
***Installing the GPU version:
conda install -c anaconda keras-gpu -y
***Installing the CPU version:
conda install -c anaconda keras -y
~~~

Installing the Synthesizability Predictor (NNCSP) package and dependencies:
~~~sh
python setup.py install
~~~


## 3. Using the model
Loading the model:
~~~python
from synthesizability import synthesizability
samples = synthesizability.get_test_samples('GaN')
samples
~~~
Output:
~~~
['NN-crystal-synthesizability-predictor/finalized_results/explore_structures/cif/GaN/GaN_9.cif',
 'NN-crystal-synthesizability-predictor/finalized_results/explore_structures/cif/GaN/GaN_12.cif']
~~~
Evaluation: (Input can be CIF files or ASE Atoms Objects)
~~~python
synthesizability.synthesizability_predictor(sample)
~~~

The output is the synthesizability likelihood of the input list:
~~~
2/2 [==============================] - 2s 783ms/step
array([0.00200533, 0.9643494 ], dtype=float32)
~~~

## 4. Reproducibility
### 4.1 Unpacking the data files
Joining the tar files:
~~~sh
cat Data.parts.a* > Data.joined.tar.gz
~~~
Untarring the files to the data folder:
~~~sh
tar -xvzf Data.joined.tar.gz
~~~
### 4.2 Data availability
All the CIF files are collected from the [Materials Project](https://materialsproject.org/), [COD](http://www.crystallography.net/cod/) and [CSPD](https://github.com/SUNCAT-Center/AtomicStructureGenerator) databases. The selected data fed to the model is available in the unwrapped data folder.

### 4.3 Preparing data from the scratch
All the used data is already prepared in the unwrapped folder. However, for reproducing everything again, follow the steps:
Set the data folders in the [config.py](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/config.py) file. 
1. Prepare the positive set:
Obtain the [COD](https://wiki.crystallography.net/howtoobtaincod/):
~~~sh
mkdir data/data_bases/
wget http://www.crystallography.net/archives/cod-cifs-mysql.tgz
tar -xvzf cod-cifs-mysql.tgz
~~~
Follow [the guideline](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/positive_data_preparation.ipynb) to create the SQL file. 

Run the entire [positive_data_preparation.ipynb](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/positive_data_preparation.ipynb) file.

2. According to the prepared guideline to prepare the CSPD data set and run [anomaly_generation.ipynb](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/anomaly_generation.ipynb) to prepare the negative set
3. Create the image files by running [data_set_selections.ipynb](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/data_set_selections.ipynb)

### 4.4 Re-training the models
1. To re-train the CNN visit: [train_cnn_d9_encoder_1_r3.ipynb](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/train_cnn_d9_encoder_1_r3.ipynb)
2. To re-train the CAE-MLP visit: [train_cae_mlp_d9_encoder_9.ipynb](https://github.com/kadkhodaei-research-group/NN-crystal-synthesizability-predictor/blob/main/train_cae_mlp_d9_encoder_9.ipynb)