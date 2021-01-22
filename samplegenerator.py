from keras.utils import Sequence
from utility.util_crystal import *
import utility.util_crystal as util
import warnings
from data_preprocess import cif_chunk2mat
from cod_tools import channel_generator


class SampleGenerator(Sequence):
    """
    This generate samples for feeding keras.fit_generator.
    It gets an idx and it decides the which batch to be passed based on the initialized file names.

    -Ali Davari
    """
    def __init__(self, filename, labels=None, batch_size=3, tot_batches=None, directory='', verbose=0, saved_chunks=True,
                 pad_len=70,
                 n_bin=128,
                 test_run=False, sub_sec=1, channels=None, name='Unnamed'):
        if channels is None:
            channels = ['atomic_number']
        self.filename, self.labels = filename, labels
        self.tot_batches = tot_batches
        self.directory = directory
        self.verbose = verbose
        self.batch_size = batch_size
        self.save_chunks = saved_chunks
        self.pad_len = pad_len
        self.bin = n_bin
        self.test = test_run
        self.current_file = None
        self.sub_sec = sub_sec
        self.channels = channels
        self.name = name
        self.sparse = True
        if self.filename[0][-3:] == 'npz':
            self.sparse = False
        if self.tot_batches is None:
            self.tot_batches = len(self.filename)
        if not self.tot_batches * self.sub_sec > 0:
            raise ValueError('You should specify total number of batches and it should be greater than 0')
        if self.test:
            print(Color.BOLD + Color.RED + '=' * 100 + Color.END, flush=True)
            txt = 'This is a test run.'
            warnings.warn(txt)
            print(txt)
            print(Color.BOLD + Color.RED + '=' * 100 + Color.END, flush=True)
        if self.verbose > 0:
            print('The generator was initialized with verbose = {}'.format(self.verbose), flush=True)

    def __len__(self):
        if self.tot_batches == 0 and self.save_chunks:
            raise ValueError("When you used saved chunks you have to specify number of total batches")

        if not self.tot_batches == 0:
            return self.tot_batches * self.sub_sec
        return int(np.ceil(len(self.filename) / float(self.batch_size)))

    def __getitem__(self, idx):
        # ######DEBUG_START
        # print('Get item: ', idx)
        # mat = np.random.rand(5, 32, 32, 32, 1)
        # return mat, mat
        # ######DEBUG-END
        try:
            time0 = datetime.now()
            if self.verbose >= 2:
                print(f'Entered generator with idx = {idx}', end='\n', flush=True)
                # print('\tMemory = {}'.format(memory_info()), end='', flush=True)
            requested_file = self.filename[int(idx / self.sub_sec)]
            requested_label = None
            if self.labels is not None:
                requested_label = self.labels[int(idx / self.sub_sec)]
            if not requested_file == self.current_file:
                if self.verbose >= 1:
                    print(f'\nReading the next file from HHD. idx = {idx} Current file = {requested_file}', flush=True)
                try:
                    util.data_mat = load_var(requested_file)
                except Exception as e:
                    red_print(f'Encountering an issue during loading: {requested_file}')
                    print(e)
                    print('Remaking the file...')
                    selected_sections = [int(requested_file.split('/')[-2])]
                    tg = requested_file.split('data_sets/')[1].split('/')[0]
                    npz_sub_sec = [int(requested_file.split('/')[-1].split('.')[0])]
                    cif_chunk2mat(total_sections=100, selected_sections=selected_sections, pad_len=self.pad_len, n_bins=self.bin,
                                  break_into_subsection=1,
                                  target_data=f'cod/data_sets/all/cif_chunks/',
                                  output_dataset=f'cod/data_sets/{tg}/mat_set_{self.bin}/', parser='ase',
                                  channels=None, make_mat_files=True, n_cpu=1, check_for_outliers=True,
                                  npz_sub_sec=npz_sub_sec, repairing=True)
                    util.data_mat = load_var(requested_file)

                if not self.sparse:
                    util.data_mat = util.data_mat[util.data_mat.files[0]]
                else:
                    util.data_mat = np.asarray(util.data_mat)
                if self.verbose >= 1:
                    print('Containing {} samples. Dividing into {} sections with ~{} samples in each'.format(
                        len(util.data_mat), self.sub_sec, int(len(util.data_mat) / self.sub_sec)))
                self.current_file = requested_file
            data = util.data_mat[np.array_split(range(len(util.data_mat)), self.sub_sec)[idx % self.sub_sec]]
            if self.test:
                warnings.warn('This is a test')
                data = data[:int(len(data) / 4)]
            if self.sparse:
                data = self.get_item_sparse(idx)
            if self.verbose >= 1:
                clc_time = str(datetime.now() - time0).split('.')[0]
                print(f'Exit generator idx={idx}, calc. time = {clc_time}', flush=True)
            if self.labels is None:
                return data, data
            else:
                return data, np.array([requested_label]*len(data))

        except Exception as e:
            save_var(locals(), 'sessions/SampleGenerator.pkl')
            write_text('sessions/SampleGenerator.txt', dict2str(locals()))
            print(dict2str(locals()))
            print('Ali: Session saved at sessions/SampleGenerator.pkl')
            print(str(e))
            raise

    def get_item_compressed(self, idx):
        file = self.filename[int(idx / self.sub_sec)]
        mat = load_var(file)
        mat = mat[mat.files[0]]
        if self.sub_sec > 1:
            mat = np.array_split(mat, self.sub_sec)[idx % self.sub_sec]
        return mat

    def get_item_sparse(self, idx):
        if self.test:
            mat = np.random.random((50, self.bin, self.bin, self.bin, len(self.channels)))
            return mat, mat
        data = bind = None
        try:
            time0 = datetime.now()
            if self.verbose >= 2:
                print(f'Entered generator with idx = {idx}', end='\n', flush=True)
            requested_file = self.filename[int(idx / self.sub_sec)]
            if not requested_file == self.current_file:
                if self.verbose >= 1:
                    print(f'\nReading the next file from HHD. idx = {idx} Current file = {requested_file}', flush=True)
                util.data_mat = np.asarray(load_var(requested_file))
                if self.verbose >= 1:
                    print('Containing {} samples. Dividing into {} sections with ~{} samples in each'.format(
                        len(util.data_mat), self.sub_sec, int(len(util.data_mat) / self.sub_sec)))
                self.current_file = requested_file
            data = util.data_mat[np.array_split(range(len(util.data_mat)), self.sub_sec)[idx % self.sub_sec]]
            if self.test:
                warnings.warn('This is a test')
                data = data[:int(len(data) / 4)]
            mat2 = np.zeros((len(data), self.bin, self.bin, self.bin, len(self.channels)))
            # print('Mat made')
            for i in range(len(data)):
                bind = data[i]['Binds']
                channels = channel_generator(data[i]['Btypes'], self.channels)
                for c in range(channels.shape[1]):
                    mat2[i, bind[:, 0], bind[:, 1], bind[:, 2], c] = channels[:, c]
        except Exception as e:
            print('Error at get_item_sparse')
            print(e)
            raise
        if self.verbose >= 1:
            clc_time = str(datetime.now() - time0).split('.')[0]
            print(f'Exit generator idx={idx}, calc. time = {clc_time}', flush=True)
        return mat2

    def __str__(self):
        out = '---\nSample Generator:'
        for key, val in self.__dict__.items():
            if val is None:
                continue
            if isinstance(val, dict):
                val = '\n'.join([str(i) + ': ' + str(j) for i, j in val.items()])
            out += key + ' = \n\t' + str(val) + '\n'
        out += '---'
        return out
