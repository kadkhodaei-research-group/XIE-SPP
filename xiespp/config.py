import os

# Your prefered data path
data_path = os.path.expanduser('Data/')
# local_data_path = os.path.expanduser('~/Local_Data/syn')
local_data_path = data_path

default_super_cell_executable = None
if default_super_cell_executable is None:
    default_super_cell_executable = os.path.expanduser('~/Apps/supercell.linux/supercell')

cod_sql_file = local_data_path + '/data_bases/cod/mysql/cod.db'

classifiers_path = 'model/classifiers/'

cspd_file = 'AtomicStructureGenerator/CSPD.db'

cnn_model_dir = 'finalized_results/cnn-3-13-9train_cnn_d9_encoder_1_r3'
cae_mlp_model_dir = 'finalized_results/train_cae_mlp_d9_encoder_9'
cae_mlp_model_clf_dir = 'finalized_results/train_cae_mlp_d9_encoder_9/mlp_clf_epoch0004'
