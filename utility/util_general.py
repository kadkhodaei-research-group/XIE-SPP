import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import sys
from os import getpid, makedirs
from os.path import expanduser
import os
from glob import glob
import subprocess
import sys
from shutil import rmtree
import pickle
from os.path import exists
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
import random
import linecache
import pandas as pd
from collections.abc import Iterable
import utility.util_plot as Plots
from config import *

# data_path = '/Volumes/ALI - WD/Data/'
# data_path = expanduser('~/Data/')
if not exists(data_path):
    data_path = '/oasis/scratch/comet/adavari/temp_project/home/adavari/Downloads/'
    if not exists(data_path):
        data_path = expanduser('/media/ali/Data/')
        if not exists(data_path + 'cod/'):
            raise FileNotFoundError('Couldn''t find any database.')
print('Data path = ', data_path, flush=True)

tot_cpu = os.cpu_count()
if os.getenv('SLURM_NTASKS_PER_NODE') is not None:
    tot_cpu = int(os.getenv('SLURM_NTASKS_PER_NODE'))

os_exists = exists


def exists(path):
    path = fix_path(path)
    return os_exists(path)


def fix_path(filename):
    if (not os_exists(filename)) and ('results' in filename) and (os_exists(f'{data_path}cod/{filename}')):
        filename = f'{data_path}cod/{filename}'
    return filename


def human_readable(num, suffix='B', output_in_mb=False):
    if output_in_mb:
        return num / 2 ** 20
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_size_of(objct):
    def get_size(obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size

    return human_readable(get_size(objct))


def memory_info(stop=True, stop_limit=10000):
    import psutil
    process = psutil.Process(getpid())
    used_mem = process.memory_info().rss
    if human_readable(used_mem, output_in_mb=True) > stop_limit and stop:
        print('Code exceeded the limit of {} MB memory.'.format(stop_limit))
        raise MemoryError('Code exceeded the limit of {} MB memory.'.format(stop_limit))
    return human_readable(used_mem)


def list_all_files(path, pattern='[0-9.]*.pkl', recursive=True, shuffle=False, error=True, random_seed=0):
    # list_all_files(data_path + 'cod/battery/cif/', pattern=r'[0-9]*/*.cif')
    if path[-1] is not '/':
        path += '/'
    path = fix_path(path)
    if not exists(path):
        raise FileExistsError(f'No such path exists. Path: {path}')
    list_of_files = sorted([f for f in glob(path + pattern, recursive=recursive)])
    if error and len(list_of_files) == 0:
        raise ValueError('No file found')
    if shuffle is True:
        random.Random(random_seed).shuffle(list_of_files)
    return list_of_files


def save_var(var_name, filename, make_path=True, compress_np=False, large_data=False):
    if make_path:
        makedirs('/'.join(filename.split('/')[:-1]) + '/', exist_ok=True)
    if filename[-3:] == 'npz':
        compress_np = True
    if compress_np:
        # if not filename[-3:] == 'npz':
        #     filename = filename + '.npz'
        np.savez_compressed(filename, var_name)
    else:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(var_name, f)
                f.close()
        except Exception as e:
            red_print(f'Large data saving: {str(e)}')
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(var_name, f, protocol=4)
                    f.close()
            except Exception as e:
                red_print(f'Error: {str(e)}')
                pass


def load_var(filename, verbose=False, uncompress=False):
    filename = fix_path(filename)
    if not exists(filename):
        filename = expanduser('~/Data/cod/') + re.split('/cod/', filename)[1]
    if not exists(filename):
        raise FileNotFoundError(filename)

    if filename[-3:] == 'npz':
        out = np.load(filename)
        if uncompress:
            out = out[out.files[0]]
    else:
        with open(filename, 'rb') as f:
            out = pickle.load(f)
    if verbose:
        print(f'File ({filename}) loaded')
    return out


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARK_CYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    # Example: print(Color.BOLD + Color.RED + 'Hello World !' + Color.END)


def red_print(txt, flush=False):
    if not isinstance(txt, str):
        txt = str(txt)
    print(Color.BOLD + Color.RED + txt + Color.END, flush=flush)


def bash_cd(command, cd):
    if not (cd == '' or cd is None):
        command = command.split('\n')
        pos = 1
        if '#!' not in command[0]:
            pos = 0
        command.insert(pos, 'cd {}'.format(cd))
        command = '\n'.join(command)
    return command


def run_bash_commands(command, wait=True, verbose=False, path=''):
    if verbose:
        command = 'set -x\n' + command
    if path is not '':
        command = bash_cd(command, path)
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=True)
    if wait:
        process.wait()
    out, err = process.communicate()
    exitcode = process.returncode
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    if verbose:
        print('-' * 10)
        print('\tstdout: ', out)
        print('\tstderr: ', err)
        print('\texit_code: ', exitcode)
        print('-' * 10)
    return out, err, exitcode


def write_text(filename, text, separator='\n', makedir=False):
    if makedir:
        makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
    if isinstance(text, list):
        text = separator.join(text)
    if not isinstance(text, str):
        text = str(text)
    with open(filename, 'w') as f:
        f.write(text)
        f.close()


def read_text(filename, line_by_line=False):
    filename = fix_path(filename)
    with open(filename, 'r') as f:
        txt = f.read()
        f.close()
    if line_by_line:
        txt = txt.split('\n')
    return txt


def save_df(df, filename):
    df.to_csv(filename, header=True, index=True, sep='\t', mode='w')


def print_df(df):
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def dict2str(d):
    return '\n'.join([str(i) + ': ' + str(j) for i, j in d.items()])


class Logger(object):
    # How to use it?
    # sys.stdout = Logger(path=self.results_path + "log.txt")
    # sys.stderr = Logger(path=self.results_path + "stderr")
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.__del__()
        del self

    def __del__(self):
        sys.stdout = self.terminal
        self.log.close()

    def __enter__(self):
        pass

    def __exit__(self, _type, _value, _traceback):
        pass


def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def get_arg_terminal(parameter, pre_ind='--', default=None):
    if not pre_ind + parameter in sys.argv:
        return default
    parameter = sys.argv[sys.argv.index(pre_ind + parameter) + 1]
    if is_number(parameter):
        parameter = float(parameter)
        if int(parameter) == parameter:
            parameter = int(parameter)
    if parameter == 'True' or parameter == 'False':
        parameter = eval(parameter)
    return parameter


def is_number(str_1):
    # Doesn't properly handle floats missing the integer part, such as ".7"
    # SIMPLE_FLOAT_REGEXP = re.compile(r'^[-+]?[0-9]+\.?[0-9]+([eE][-+]?[0-9]+)?$')
    # Example "-12.34E+56"      # sign (-)
    #     integer (12)
    #           mantissa (34)
    #                    exponent (E+56)

    # Should handle all floats
    # FLOAT_REGEXP = re.compile(r'^[-+]?([0-9]+|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?$') # I changed it to following
    FLOAT_REGEXP = re.compile(r'^[-+]?([0-9]+\.?|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?$')
    # Example "-12.34E+56"      # sign (-)
    #     integer (12)
    #           OR
    #             int/mantissa (12.34)
    #                            exponent (E+56)

    # return True if FLOAT_REGEXP.match(str_1) else False
    return bool(FLOAT_REGEXP.match(str_1))


def find_all_numbers(str_1):
    NUMBERS = re.compile(r'[-+]?([0-9]+\.?|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?')
    return [i.regs[0] for i in NUMBERS.finditer(str_1)]


def run_in_parallel(function, iterable, n_jobs=2, method=1):
    from joblib import Parallel, delayed
    output = None
    if method == 1:
        output = Parallel(n_jobs=n_jobs)(delayed(function)(it) for it in iterable)

    if method == 2:
        output = Parallel(n_jobs=n_jobs, verbose=1, backend="threading")(
            map(delayed(function), iterable))

    from multiprocessing import Pool
    if method == 3:
        print('multiprocessing!')
        p = Pool(processes=n_jobs)
        output = p.map(function, iterable)
        p.close()
        p.join()

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import ProcessPoolExecutor
    if method == 4:
        print('multiprocessing!')
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            responses = executor.map(function, iterable)
        output = list[responses]

    if method == 5:
        print('Multi-threading')
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            responses = executor.map(function, iterable)
        output = list[responses]

    return output


# def classification_accuracy(y, y_prediction, threshold=0.5):
#     from sklearn.metrics import accuracy_score
#     ypre
#     accuracy_score()