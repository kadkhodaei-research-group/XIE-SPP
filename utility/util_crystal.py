from utility.util_general import *
import sys
from os import makedirs
from shutil import rmtree
import pickle
import numpy as np
import gc
import pandas
import re
import sqlite3
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime
import shutil
import platform
from ase.visualize import view as ase_view
from socket import gethostname
import bokeh.sampledata.periodic_table as pt

data_mat = None
if data_mat == 1:  # To remove unused packages warnings
    warnings.simplefilter("ignore")
    plt.figure()
    json.load(data_mat)
gc.enable()


def change_chunks(input_path, output_path, sections, constraints=None):
    print('Changing chunks from: ', input_path)
    if not output_path[-1] == '/':
        output_path += '/'
    print('To: ', output_path)
    if not exists(output_path):
        makedirs(output_path, exist_ok=True)
    df_summary = summary(input_path)
    file_sizes = df_summary['n_samples']
    input_files = list(df_summary['filename'])
    sections = np.array_split(range(sum(file_sizes)), sections)
    loaded_files = []
    data = []
    sec_itr = 0
    df = pandas.DataFrame(columns=['filename', 'n_samples'])
    while True:
        if len(loaded_files) == 0:
            if len(input_files) == 0:
                break
            print('Loading the next input file: ', input_files[0])
            loaded_files = load_var(input_files.pop(0))
            if len(loaded_files) == 0:
                continue
            # with open(input_files.pop(0), 'rb') as file:
            #     loaded_files = pickle.load(file)
        tmp = loaded_files.pop(0)

        if np.asarray(constraints == re.split('/', tmp['filename'])[-1][:-4]).any() or constraints is None:
            data.append(tmp)
        if len(data) == len(sections[0]):
            print('Saving section #{} with {} elements inside.'.format(sec_itr, len(data)))
            sections.pop(0)
            file = output_path + '{:04d}.pkl'.format(sec_itr)
            save_var(data, file, make_path=True)
            df = df.append({'filename': file, 'n_samples': len(data)}, ignore_index=True)
            # with open(output_path + '{:04d}.pkl'.format(sec_itr), 'wb') as file:
            #     pickle.dump(data, file)
            sec_itr += 1
            del data
            data = []
            gc.collect()
    df['CUM_SUM'] = df['n_samples'].cumsum()
    save_var(df, output_path + 'summary.pkl')
    with open(output_path + 'summary.txt', 'w') as f1:
        print(df.to_string(), file=f1)
    summary(output_path)


def summary(chunks_path, pattern='[0-9.]*.pkl'):
    def make_summary(input_path):
        print(Color.BOLD + Color.RED + '=' * 100 + Color.END, flush=True)
        print('Making a summary of: ', input_path)
        files = list_all_files(input_path, pattern=pattern)
        print(len(files), 'files were found.')
        df = pandas.DataFrame(columns=['filename', 'n_samples'])
        for i in range(len(files)):
            file = files[i]
            # if i in np.floor(np.linspace(0, len(files) - 1, 10)):
            print('{}/{} completed.\t'.format(i, len(files)), file)
            with open(file, 'rb') as f1:
                data = pickle.load(f1)
            df = df.append({'filename': file, 'n_samples': len(data)}, ignore_index=True)
        df['CUM_SUM'] = df['n_samples'].cumsum()
        with pandas.option_context('display.max_rows', None, 'display.max_columns',
                                   None):  # more options can be specified also
            print(df)
        with open(input_path + 'summary.pkl', 'wb') as f1:
            pickle.dump(df, f1)
        save_df(df, input_path + 'summary.txt')
        return df

    if not exists(chunks_path + 'summary.pkl'):
        df_summary = make_summary(chunks_path)
    else:
        df_summary = load_var(chunks_path + 'summary.pkl')
        # with open(chunks_path + 'summary.pkl', 'rb') as f:
        #     df_summary = pickle.load(f)
        if not len(df_summary) == len(list_all_files(chunks_path, pattern=pattern)):
            df_summary = make_summary(chunks_path)
        if not re.split('/cod/', df_summary['filename'][0])[0] + '/' == data_path:
            df_summary['filename'] = df_summary['filename'].str.replace(
                re.split('/cod/', df_summary['filename'][0])[0] + '/', data_path)
            save_var(df_summary, chunks_path + 'summary.pkl')
            save_df(df_summary, chunks_path + 'summary.txt')
    return df_summary


def run_query(query, root=data_path, db_path='cod/mysql/cod.db', make_table=False):
    def regexp(expr, item):
        reg = re.compile(expr)
        return reg.search(item) is not None

    conn = sqlite3.connect(root + db_path)
    conn.create_function("REGEXP", 2, regexp)
    c = conn.cursor()
    c.execute(query)
    out = c.fetchall()
    if make_table:
        col = ' '.join(
            re.split(',| |, ', query[query.lower().find('select') + len('select'):query.lower().find('from')])).split()
        if len(col) == 0:
            raise ValueError('Was not able to make the table out of the requested query')
        out = pandas.DataFrame(out, columns=col)
    return out


def include_acceptable_elements(atoms, exclusions=None, inclusion=None):
    types_of_elements = np.unique(pt.elements.metal)
    all_elements = np.asarray(pt.elements.symbol)
    if inclusion is None:
        inclusion = types_of_elements
    acceptable_elements = pandas.Series()
    for i in inclusion:
        if i == 'nobel gas' or i == 'gas':
            i = 'noble gas'  # it was a but in pd
        if i in types_of_elements:
            acceptable_elements = acceptable_elements.append(pt.elements.symbol[pt.elements.metal == i])
        elif i in all_elements:
            acceptable_elements = acceptable_elements.append(pt.elements.symbol[pt.elements.symbol == i])
        else:
            raise ValueError('Could not find the element in the periodic table')
    for i in exclusions:
        if i == 'nobel gas' or i == 'gas':
            i = 'noble gas'  # it was a but in pd
        if i in types_of_elements:
            removed_elements = pt.elements.symbol[pt.elements.metal == i]
        elif i in all_elements:
            removed_elements = pt.elements.symbol[pt.elements.symbol == i]
        else:
            raise ValueError('Could not find the element in the periodic table')
        acceptable_elements = acceptable_elements.append(removed_elements).drop_duplicates(keep=False)
        if len(acceptable_elements) < 1:
            raise ValueError('There is no acceptable elements!')
    if set(np.unique(atoms.get_chemical_symbols())).issubset(acceptable_elements):
        return True
    else:
        return False


class RunSet:
    def __init__(self, params=None, ini_from_path=None, new_result_path=True, verbose=True, log=True):
        if params is None:
            params = {}
        skip_error = False
        print(f'Starting time: {str(datetime.now())[:19]}')
        if ini_from_path is not None:
            if verbose:
                print(f'Initializing from {ini_from_path}')
            if ini_from_path[-1] == '/':
                ini_from_path += 'run.pkl'
            skip_error = params.get('skip_error', False)
            run_set = load_var(ini_from_path)
            self.__dict__.update(run_set.__dict__.copy())
            run_set.params.update(params.copy())
            params = run_set.params
            self.__dict__.update(params.copy())
            self.last_run = '/'.join(ini_from_path.split('/')[:-1]) + '/'
            if new_result_path:
                self.results_path = None
                self.run_id = None
                params.pop('run_id', None)
            # else:
            #     params.update({'run_id': })
            # params['run_id'] = self.run_id
        elif type(params) is RunSet:
            self.__dict__.update(params.__dict__.copy())
            params = params.params
        else:
            self.pad_len = params.pop('pad_len', None)  # return None if key isn't in the dict
            self.n_bins = params.pop('n_bins', None)
            self.sec = params.pop('sec', None)
            self.file_sub = params.pop('file_sub', None)
            self.num_epochs = params.pop('num_epochs', None)
            self.batch_sub_sec = params.pop('batch_sub_sec', None)
            self.tot_samples_df = self.test_x = self.train_x = None
            self.train_generator = self.test_generator = None
            self.file_suffix = None
        self.argv = sys.argv
        self.computer_name = gethostname()
        self.run_start_time = datetime.now()
        self.run_time_duration = None
        self.lap_time = datetime.now()
        self.run_end_time = None
        self.timer = [self.run_start_time, None, None]
        self.params = params
        self.plots = None
        self.random_state = params.get('random_seed', 0)
        chunks_path = data_path + 'cod/data_sets/' + params.get('data_set', '')
        if not exists(chunks_path):
            chunks_path = data_path + params['data_set']
            if (not exists(chunks_path)) and (not skip_error):
                raise FileExistsError(f'Could not find the chunk path. {chunks_path}')
        self.chunks_path = chunks_path
        self.job_id = get_arg_terminal('job_id')
        while True:
            try:
                self.find_run_id(params)
                break
            except FileExistsError as e:
                red_print(e)
                if get_arg_terminal('results') is not None:
                    raise
                import time
                time.sleep(random.Random().random() * 10)
                self.run_id = None
        import tarfile
        with tarfile.open(self.results_path + 'codes.tar.gz', "w:gz") as tar:
            for f in list_all_files('./', pattern='*.py', recursive=False):
                tar.add(f, arcname=os.path.basename(f))
        if log:
            sys.stdout = Logger(path=self.results_path + "log.txt")
            sys.stderr = Logger(path=self.results_path + "stderr")
        print(f'Main file: {sys.argv[0]}')
        print(f'Results path: {self.results_path}')
        if 'test' in self.params.keys():
            warnings.warn('This is a test run with smallest database')
            red_print('This is a test run with smallest database')
        if ini_from_path is not None:
            if verbose:
                print(f'Initializing from {ini_from_path}')

    def __str__(self):
        out = ''
        for key, val in self.__dict__.items():
            if key is 'train_x' or key is 'test_x' or '_generator' in key:
                continue
            if val is None:
                continue
            if isinstance(val, dict):
                val = '\n'.join([str(i) + ': ' + str(j) for i, j in val.items()])
            out += key + ' = \n\t' + str(val) + '\n'
        return out

    def timer_lap(self):
        now = datetime.now()
        lap = now - self.lap_time
        print('Lap time: {}'.format(str(lap).split('.')[0]), flush=True)
        self.lap_time = now
        return lap

    def end(self):
        self.run_end_time = datetime.now()
        print(f'End time: {self.run_end_time}')
        self.run_time_duration = self.run_end_time - self.run_start_time
        print('The run file is being deleted. Saving run file to: ' + f'{self.results_path}run.pkl')
        save_var(self, f'{self.results_path}run.pkl')
        write_text(f'{self.results_path}run.txt', self)
        print('Duration: {}\nThe End'.format(str(self.run_time_duration).split('.')[0]), flush=True)

    def find_run_id(self, params):
        if get_arg_terminal('results') is not None:
            params['run_id'] = expanduser(get_arg_terminal('results').strip('\"').strip('\''))
        self.run_id = params.get('run_id', None)
        if self.run_id is None:  # Finding the next id spot
            previous_runs = list_all_files('results/', pattern='run_*', recursive=False, error=False)
            if len(previous_runs) == 0:
                previous_runs = ['run_0']
            self.run_id = int(re.findall("[0-9]+", re.split('/', previous_runs[-1])[-1])[-1]) + 1
        if isinstance(self.run_id, str):
            if not self.run_id[0] == '/':
                self.results_path = 'results/' + self.run_id
            else:
                if not self.run_id[-1] == '/':
                    self.run_id = self.run_id + '/'
                self.results_path = self.run_id
        else:
            self.results_path = 'results/run_{:03d}/'.format(self.run_id)
        makedirs(self.results_path, exist_ok=True if 'run_id' in params.keys() else False)

    def save_plot(self, filename=None, show=True, verbose=True, transparent=False,
                  over_write=False, to_one_pdf=False, formats=None):
        if formats is None:
            formats = ['png', 'pdf']
        if to_one_pdf:
            if self.plots is None:
                from matplotlib.backends.backend_pdf import PdfPages
                self.plots = PdfPages(f'{self.results_path}plots.pdf')
                Plots.plot_save(self.plots, formats='pdf')
        if (filename is not None) and (not over_write):
            previous_plots = list_all_files(self.results_path, pattern=f'{filename}*.png', recursive=False, error=False)
            if len(previous_plots) > 0:
                filename += f'_{len(previous_plots) + 1:03}'
        if filename is None:
            previous_plots = list_all_files(self.results_path, pattern='plot_*', recursive=False, error=False)
            if len(previous_plots) == 0:
                previous_plots = ['run_0']
            pl_id = int(re.findall("[0-9]+", re.split('/', previous_plots[-1])[-1])[-1]) + 1
            filename = f'plot_{pl_id:03}'
        Plots.plot_save(filename, path=self.results_path, formats=formats, save_to_papers=False,
                        transparent=transparent)
        if verbose:
            print(f'Plot saved to {filename}.')
        if show:
            plt.show()


def view_atom_vmd(atom, viewer='VMD'):
    ase_view(atom, viewer=viewer)


def materials_project_cif(mp_id, ticket="9TaFAWEusVChlUdH", verbose=True, skip_error=True):
    def get_data(mp_id_1):
        import pymatgen as mg
        mp = mg.MPRester(ticket)
        data = mp.get_data(mp_id_1)
        if len(data) == 0:
            raise FileNotFoundError('No object found on Material Project!')
        return data

    if isinstance(mp_id, list):
        df = pd.DataFrame()
        for i in range(len(mp_id)):
            if verbose:
                print(f'Getting #{i}, id: {mp_id[i]}')
            try:
                entry = get_data(mp_id[i])
            except Exception as e:
                red_print(f'Error in reading {mp_id[i]}: {e}')
                continue
            for d in entry:
                if len(df) == 0:
                    df = pd.DataFrame([d], index=[0])
                else:
                    df = df.append(d, ignore_index=True)
        return df
    return get_data(mp_id)


class PeriodicTable:
    def __init__(self):
        import bokeh.sampledata.periodic_table as pt
        self.table = pt.elements.copy()
        self.table['atomic mass'] = [float(re.findall('[0-9.]+', i)[0]) for i in self.table['atomic mass']]


if __name__ == "__main__":
    PeriodicTable()
    print('end')
