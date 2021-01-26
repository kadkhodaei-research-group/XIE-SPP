# Neural Network Crystal Synthesizability Predictor

### 1.Abstract
Predicting the synthesizability of hypothetical crystals has proven to be challenging due to the wide range of parameters that govern the crystalline materials synthesis. Yet exploring the exponentially large space of novel crystals for any future application demands an accurate predictive capability for synthesis likelihood to avoid the haphazard trial-and-error. While prevalent benchmarks of synthesizability rely on the energetics of crystal structures, we take an alternative approach to select features of synthesizability from the latent information embedded in existing crystalline materials. We convert the atomic structures of known synthesized or observed crystals in databases into three-dimensional pixel-wise images color-coded by their chemical attributes and use them for training a neural-network convolutional autoencoder (CAE). We extract the latent features of synthesizability hidden in structural and chemical arrangements of crystalline materials from the auto-encoder. The accurate classification of materials into synthesizable crystals vs. crystal anomalies based on these features across a broad range of crystal structure types and chemical compositions confirmed our model's validity. The usefulness of the model is illustrated by predicting the synthesizability of hypothetical candidates for battery electrodes and thermoelectric applications.

### 2. Examples of usage
*	Evaluating the synthesis likelihood
Consider a list of files for evaluation; refer to ASE documentation for the list of compatible types.
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

Follow the instruction on the [conda's website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or follow according to this method, tested on an ubuntu-20 machine:
```bash
$ cd ~

$ wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
$ bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ~/anaconda
$ rm Anaconda3-4.2.0-Linux-x86_64.sh
$ echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 

$ source ~/.bashrc
$ conda update conda
```
Create an envirenment for this package (It's is possible to use other envirnments if you already have one ready) and install all the [minimum requirements](minimum_requirements.txt) to run this model.

```bash
$ cd TO/THE/PATH/WHERE/YOU/KEEP/NN-crystal-synthesizability-predictor

$ conda create -n synthesizability python=3.7
$ conda activate synthesizability
$ pip install -r minimum_requirements.txt
```
This install all the packages needed to compute the synthesizability likelihood of crystals. These packages are:

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
If you wish to visualize atoms as you in the test file:
```bash
$ conda install nglview -c bioconda
```

### 4. An explanation of the code structure
The belows chart displays the four different main part of the code. 

**config.py :** Includes some global parameters and paths
#### 4.1. Data preparation 
Includes two major components to prepare the positive and the negatice classes.

* __data_preprocess_positive.py__
> **prepare_data_from_scratch:** It prepares all the positive data from scratch. If one of the processors is being killed the command line prompts
    that are being run by run_bash_commands can be directly use in the command line in the the same directory.
    
> **filter_cif_with_partial_occupancy:** It goes over the COD crystals and saves a list of structures with a partial occupancy to supercell_list.sh
    
> **cif2chunk:** It get a directory of CIF files and it parses all the CIF to ASE Atoms Object and save them
    into chunks of pickle data sets.
    
> **cif_chunk2mat:** It converts a list of chunks of CIF files to SPARSE 3D cubic images of crystals.

> **cif_chunk2mat_helper:** The helper function to cif_chunk2mat for parallelization.

> **load_data:** Loads the crystal space representation of whole database and prepares test and train sets, but it doesn't do
    any processes like SS, PCA or Over-Sampling

> **data_preparation:** Loads the train and test sets and it applies Standard Scalar, PCA and OverSampling.

* __cspd_anomaly_generator.py__
> **list_all_the_formulas:** Creating a table of all the compositions available in the Literature and their number of repetitions
    (The mat2vec package borrowed from [here](https://github.com/materialsintelligence/mat2vec))
    
> **anomaly_gen_lit_based:** Creates crystal anomaly by generating hypothetical structures of well studied crystals and removing known structure
    from the COD. ([Helper package](https://github.com/SUNCAT-Center/AtomicStructureGenerator))
    
> **cspd_hypotheticals:** Generate all the possible hypothetical crystal structures given the input elements.



#### 4.2. 

### 5. Re-training the model from the scratch (This section is partially completed)
#### 5.1. How to obtain required data bases and packages
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
