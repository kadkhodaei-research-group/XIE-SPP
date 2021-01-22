import threading
from utility.util_general import *
import urllib.request


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            print('Thread started')
            self.process = subprocess.Popen(self.cmd, shell=True)
            self.process.communicate()
            print('Thread finished')

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            print('Terminating process')
            self.process.terminate()
            thread.join()
        print(self.process.returncode)


def prepare_supercell(verbose=True):
    if verbose:
        print('Preparing the SuperCell Program', flush=True)
    if not exists(data_path + 'cod/'):
        raise FileNotFoundError(f'{data_path} + cod/')
    # executable = f'{data_path}cod/supercell.linux/supercell'
    executable = default_super_cell_executable
    if exists(executable):
        return executable
    print(f'Downloading the Supercell Program to: {data_path}cod/supercell-linux.tar.gz')
    url = 'https://orex.github.io/supercell/exe/supercell-linux.tar.gz'
    urllib.request.urlretrieve(url, data_path + '/supercell-linux.tar.gz')
    run_bash_commands('tar -zxvf supercell-linux.tar.gz', path=data_path+'cod/', verbose=True, wait=True)
    return executable


def run_super_cell_program(super_cell_executable=None, verbose=True):
    if verbose:
        print('Running the Supercell Program to handle the occupancies')
    if super_cell_executable is None:
        super_cell_executable = expanduser('~/Apps/supercell.linux/supercell')
        if not exists(super_cell_executable):
            super_cell_executable = prepare_supercell()

    with open('supercell_list.sh', 'r') as f:
        file = f.readlines()
        f.close()

    for i in range(len(file)):
        out_path = data_path + '/'.join(re.split('-o', file[i])[1].split('/')[-6:-1]) + '/'
        out_path = out_path.split('/')
        out_path[-5] = out_path[-5] + '_2'
        out_path = '/'.join(out_path)
        in_file = data_path + '/'.join(re.split('-i', re.split('-o', file[i])[0])[1].split('/')[-6:])
        out_file = out_path + in_file.split('/')[-1][:-5] + '_super.cif'
        makedirs(out_path, exist_ok=True)
        print(i, out_file)
        c = Command(f'{super_cell_executable} -i {in_file} -o {out_file} -n r1 -v 0')
        c.run(timeout=5)


if __name__ == "__main__":
    run_super_cell_program()
    print('End')
