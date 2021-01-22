# Neural Network Crystal Synthesizability Predictor

### 1.Abstract
Predicting the synthesizability of hypothetical crystals has proven to be challenging due to the wide range of parameters that govern the crystalline materials synthesis. Yet exploring the exponentially large space of novel crystals for any future application demands an accurate predictive capability for synthesis likelihood to avoid the haphazard trial-and-error. While prevalent benchmarks of synthesizability rely on the energetics of crystal structures, we take an alternative approach to select features of synthesizability from the latent information embedded in existing crystalline materials. We convert the atomic structures of known synthesized or observed crystals in databases into three-dimensional pixel-wise images color-coded by their chemical attributes and use them to train a neural-network convolutional autoencoder (CAE). We extract the latent features of synthesizability hidden in structural and chemical arrangements of crystalline materials form the auto-encoder. The accurate classification of materials into synthesizable crystals vs. crystal anomalies based on these features across a broad range of crystal structure types and chemical compositions confirmed the validity of our model. The usefulness of the model is illustrated by predicting the synthesizability of hypothetical candidates for battery electrodes and thermoelectric applications.

### 2. Examples of usage
*	Evaluating the synthesis likelihood
Consider a list of files for evaluation, refer to ASE documentation for the list of compatible types
```python
from predict_synthesis import predict_crystal_synthesis

crystal_structures_files = [
'test_electrode_materials_samples/cif/003/mp-25829.cif',
'5910201.cif',
'1000011.cif'
]
predict_crystal_synthesis(crystal_structures_files)

# Or for other types of input e.g. POSCAR

predict_crystal_synthesis('path/to/POSCAR', format='vasp')

# Or ASE atoms object

predict_crystal_synthesis(atoms)
```
* For a more detailed example visit [here](testing_synthesizability_predictor.ipynb)


### 3. Instaling pre-requisits to evaluate synthesizability likelihood
* Installing conda (if you don't have it)

Follow the instruction on the [conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or follow according to this tested method on ubuntu 20:
```bash
$ cd ~

$ wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
$ bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ~/anaconda
$ rm Anaconda3-4.2.0-Linux-x86_64.sh
$ echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 

$ source ~/.bashrc
$ conda update conda
```
Create an envirenment for this package (It's is possible to use other envirnments if you already have one ready)

```bash
$ cd TO/THE/PATH/WHERE/YOU/KEEP/NN-crystal-synthesizability-predictor

$ conda create -n synthesizability python=3.7
$ conda activate synthesizability
$ pip install -r minimum_requirements.txt
```
This install all the packages needed to compute the synthesizability likelihood. These packages are:

```
ase==3.17.0
Keras==2.1.6
pandas==0.24.2
seaborn==0.10.0
scikit-learn==0.23.2
tensorflow==1.14.0
jupyter
jupyterlab
bokeh==1.2.0
matplotlib==3.1.3
h5py==2.10.0
```
If you wish to visualize atoms as you run the test file:
```bash
$ conda install nglview -c bioconda
```

### 4. Re-training the model from the scratch
#### 4.1 How to obtain required data bases and packages
Considering a path for all the necessary packages and databases. This is separate from when you install the NN Crystal Synthesizability Predictor package (Requires 200 GB)
```bash
$ export DATA_PATH=/YOUR/CHOICE/OF/PATH/
$ cd $DATA_PATH
```
Obtraining the COD database
```bash
$ mkdir cod
$ cd $DATA_PATH/cod 
$ wget http://www.crystallography.net/archives/2019/data/cod-rev214634-2019.04.15.tgz
$ tar -zxvf cod-rev214634-2019.04.15.tgz
```
You should have two directories of cif/ and mysql/ created at $DATA_PATH/cod
Follow the [COD instruction](https://wiki.crystallography.net/creatingSQLdatabase/) to create a SQL database in the mysql/ directory. 

Preparing the supercell program:
```bash
$ cd $DATA_PATH
$ wget https://orex.github.io/supercell/exe/supercell-linux.tar.gz
$ tar -zxvf supercell-linux.tar.gz
```
Now the Supercell executable should be at: $DATA_PATH/supercell.linux/supercell
