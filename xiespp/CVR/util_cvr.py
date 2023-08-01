from collections.abc import Iterable
import itertools
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import tqdm as tqdm_root
import contextlib
import os


tot_cpu = os.cpu_count()
if os.getenv('SLURM_NTASKS_PER_NODE') is not None:
    tot_cpu = int(os.getenv('SLURM_NTASKS_PER_NODE'))


def save_var(var_name, filename, make_path=True, verbose=False):
    filename = Path(filename)
    if make_path:
        filename.parents[0].mkdir(parents=True, exist_ok=True)
    if verbose:
        print('Saving to: ', filename)
    try:
        with open(filename, 'wb') as f:
            pickle.dump(var_name, f)
            f.close()
    except Exception as e:
        print(f'Large data saving error: {str(e)}')
        try:
            with open(filename, 'wb') as f:
                pickle.dump(var_name, f, protocol=4)
                f.close()
                print('Large data saved with protocol 4.')
        except Exception as e:
            print(f'Error: {str(e)}')
            pass


class ColorPrint:
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
    def red_print(self, txt, flush=True):
        if not isinstance(txt, str):
            txt = str(txt)
        print(self.BOLD + self.RED + txt + self.END, flush=flush)


def np_choose_optimal_dtype(arr, return_dtype=False):
    """
    Return the optimal dtype for a numpy array.
    """
    assert np.array_equal(np.floor(arr), arr), 'np array must be integer'
    min_val = np.min(arr)
    max_val = np.max(arr)
    type_list = [np.uint8, np.uint16, np.uint32, np.uint64]
    if min_val < 0:
        type_list = [np.int8, np.int16, np.int32, np.int64]
    for d_type in type_list:
        if np.iinfo(d_type).min <= min_val and np.iinfo(d_type).max >= max_val:
            if return_dtype:
                return d_type
            return np.array(arr, dtype=d_type)

    raise ValueError('Could not find a dtype for the array.')


@contextlib.contextmanager
def mute_print(enabled=True):
    """
    A context manager that optionally redirects stdout to a dummy file.
    Usage:
        with mute_print():
            print('This will not be printed')
        or
        with mute_print(verbose=True):
            print('This will be printed')
    """
    if not enabled:
        yield
    else:
        import io
        with io.StringIO() as dummy_file:
            with contextlib.redirect_stdout(dummy_file):
                yield


@contextlib.contextmanager
def mute_warnings(enabled=True):
    """
    A context manager that temporarily suppresses warnings.
    Usage:
        with mute_warnings():
            warnings.warn('This will not be printed')
    """
    if enabled:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Add this block to configure logging for the pint library
            import logging
            pint_logger = logging.getLogger('pint')
            previous_level = pint_logger.getEffectiveLevel()
            pint_logger.setLevel(logging.ERROR)

            yield

            pint_logger.setLevel(previous_level)  # Reset the logging level
    else:
        yield


def parallel_apply(func, *args, n_jobs=tot_cpu, ignor_warnings=True, progres_bar=True, parallel=True,
                   mute_printing=True, **kwargs):
    """
    Apply a function to a pandas Series (Or similar iterables) in parallel.
    """
    from joblib import Parallel, delayed

    # Swapping func and args if func is not a function
    if not hasattr(func, '__call__'):
        tmp = func
        func = args[0]
        args = (tmp,)

    iterator = zip(*args)
    if progres_bar:
        iterator = pbar(iterator, total=len(args[0]))

    if n_jobs == 1:
        parallel = False
    if n_jobs == -1:
        n_jobs = tot_cpu

    with mute_warnings(enabled=ignor_warnings):
        with mute_print(enabled=mute_printing):
            if not parallel:
                return [func(*it, **kwargs) for it in iterator]

            output = Parallel(
                n_jobs=n_jobs,
                # prefer='threads' if n_jobs == 1 else 'processes',
            )(delayed(func)(*it, **kwargs) for it in iterator)
            # output = parallel_pool(delayed(func)(it) for it in pbar(df))

    return output


def pbar(*args, verbose=True, **kwargs):
    if not verbose:
        return args[0]

    if 'position' not in kwargs:
        kwargs['position'] = 0
    if 'leave' not in kwargs:
        kwargs['leave'] = True
    if 'total' not in kwargs:
        # noinspection PyBroadException
        try:
            kwargs['total'] = len(args[0])
        except Exception:
            pass
    if 'total' not in kwargs:
        # noinspection PyBroadException
        try:
            kwargs['total'] = len(args[0].gi_frame.f_locals['self'])
        except Exception:
            pass
    return tqdm_root.tqdm(*args, **kwargs)


def prime_factors(n):
    """
    Return the prime factors of a number.
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
