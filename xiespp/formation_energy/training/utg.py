from pathlib import Path
from glob import glob
import os
import warnings


tot_cpu = os.cpu_count()
if os.getenv('SLURM_NTASKS_PER_NODE') is not None:
    tot_cpu = int(os.getenv('SLURM_NTASKS_PER_NODE'))


def pbar(*args, verbose=True, **kwargs):
    if not verbose:
        return args[0]
    try:
        import tqdm as tqdm_root
    except ImportError:
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


def list_all_files(path, pattern='*', recursive=True, shuffle=False, error=True, random_seed=0):
    list_of_files = sorted([f for f in glob(str(Path(path, pattern)), recursive=recursive)])
    if error and len(list_of_files) == 0:
        raise ValueError('No file found')
    return list_of_files

def parallel_apply(func, *args, n_jobs=tot_cpu, ignor_warnings=True, progres_bar=True, parallel=True,
                   mute_printing=True, **kwargs):
    """
    Apply a function to a pandas Series (Or similar iterables) in parallel.
    """
    from joblib import Parallel, delayed
    # import pandas as pd
    # import numpy as np
    # from tqdm import tqdm

    # # Split the dataframe into chunks
    # chunks = np.array_split(df, n_jobs)
    # # Apply the function to each chunk
    # results = Parallel(n_jobs=n_jobs)(delayed(func)(chunk, **kwargs) for chunk in pbar(chunks))
    # Concatenate the results
    # return pd.concat(results)

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not parallel:
            return [func(*it, **kwargs) for it in iterator]

        output = Parallel(
            n_jobs=n_jobs,
            # prefer='threads' if n_jobs == 1 else 'processes',
        )(delayed(func)(*it, **kwargs) for it in iterator)
        # output = parallel_pool(delayed(func)(it) for it in pbar(df))

    return output