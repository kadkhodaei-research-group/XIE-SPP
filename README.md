# Neural Network Crystal Synthesizability Predictor

## 1.Abstract
Predicting the synthesizability of hypothetical crystals has proven to be challenging due to the wide range of parameters that govern the crystalline materials synthesis. Yet exploring the exponentially large space of novel crystals for any future application demands an accurate predictive capability for synthesis likelihood to avoid the haphazard trial-and-error. While prevalent benchmarks of synthesizability rely on the energetics of crystal structures, we take an alternative approach to select features of synthesizability from the latent information embedded in existing crystalline materials. We convert the atomic structures of known synthesized or observed crystals in databases into three-dimensional pixel-wise images color-coded by their chemical attributes and use them to train a neural-network convolutional autoencoder (CAE). We extract the latent features of synthesizability hidden in structural and chemical arrangements of crystalline materials form the auto-encoder. The accurate classification of materials into synthesizable crystals vs. crystal anomalies based on these features across a broad range of crystal structure types and chemical compositions confirmed the validity of our model. The usefulness of the model is illustrated by predicting the synthesizability of hypothetical candidates for battery electrodes and thermoelectric applications.

## 2. Examples of usage
*	Evaluating the synthesis likelihood
Consider a list of files for evaluation, refer to ASE documentation for the list of compatible types
```python
# from predict_synthesis import predict_crystal_synthesis

crystal_structures = [
'5910201.cif',
'POSCAR'
]
predict_crystal_synthesis(crystal_structures)
```

## 3. Instaling pre-requisits to evaluate synthesizability likelihood

## 4. Re-training the model from the scratch
### 4.1 How to obtain required data bases and packages
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
