import numpy as np
from keras.utils import Sequence

from typing import Generator, Tuple
from mytypes import Filename, NDArray

class RechitHandler(Sequence):
    def __init__(self, file: Filename, other_inputs: NDArray, labels: NDArray, weights: NDArray, 
                 batch_size: int, image_size: int, which_set: str = 'train') -> None:
        """set needs to be "train", "val" or "test""""
        if which_set=='train': use_data = slice(0, int(0.6*len(labels)))
        elif which_set=='val': use_data = slice(int(0.6*len(labels)), int(0.8*len(labels)))
        elif which_set=='test': use_data = slice(int(0.8*len(labels)), None)
        else: raise ValueError('set needs to be "train", "val" or "test')

        self.other_inputs = other_inputs[use_data]  # 2d array
        self.labels = labels[use_data]  # 1d array
        self.weights = weights[use_data]
        self.batch_size = batch_size
        self.image_size = image_size

        self.batch_shape = (batch_size, image_size, image_size)
        sparse_rechits = np.load(file)  # this is a dict with 4 entries
        self.values = sparse_rechits['values'][use_data]
        self.idx_photon, self.idx_row, self.idx_col = tuple(sparse_rechits.values())[1:]
        self.idx_photon = self.idx_photon[use_data]
        self.idx_row = self.idx_row[use_data]
        self.idx_col = self.idx_col[use_data]
        
        # idx_photon indexes which photon the pixel belongs to
        # idx_row and idx_col are the idxs of the pixel in the image

        self.num_photons = self.idx_photon[-1] + 1
        self.num_batches = self.num_photons//batch_size + 1
        self.slices = self.get_slices()

        # TODO shuffle data
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, index):
        this_slice = self.slices[index]
        values = self.values[this_slice]
        shifted_photon_index = self.idx_photon[this_slice] - index*self.batch_size # make sure the first index of the first array is always 0
        indices = (shifted_photon_index, self.idx_row[this_slice], self.idx_col[this_slice])
        dense_rechits = self.sparse_to_dense(values, indices)

        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batched_inputs = self.other_inputs[:, batch_slice]
        batched_labels = self.labels[batch_slice]
        batched_weigths = self.weights[batch_slice]
        return [dense_rechits, batched_inputs], batched_labels, batched_weigths


    def shuffle_data(self):
        pass
    
    def get_slices(self):  
        photon_idxs = np.unique(self.idx_photon, return_index=True)[1]
        # photon_idxs are the indices of the first appearance of each number in idxs_photons
        slices = [slice(photon_idxs[i*self.batch_size], photon_idxs[(i+1)*self.batch_size]) for i in range(self.num_batches-1)]
        slices += [slice(slices[-1].stop, len(self.idx_photon))]  # last batch is usually smaller
        return slices

    def sparse_to_dense(self, values: NDArray, indices: Tuple[NDArray, NDArray, NDArray]) -> NDArray:
        dense = np.zeros(self.batch_shape, dtype=np.float32)
        dense[indices] = values
        return dense

    # def batch_generator(self) -> Generator[NDArray, None, None]:
    #     for i, this_slice in enumerate(self.slices):
    #         values = self.values[this_slice]
    #         shifted_photon_index = self.idx_photon[this_slice] - i*self.batch_size # make sure the first index of the first array is always 0
    #         indices = (shifted_photon_index, self.idx_row[this_slice], self.idx_col[this_slice])
    #         yield self.sparse_to_dense(values, indices)

import pandas as pd
from time import time
small_file = '/home/home1/institut_3a/kappe/work/data/test_rechit_format/original_sparse.npz'
large_file = 'data/rechits_11x11_sparse.npz'
# Handler = RechitHandler(small_file, batch_size=4096, image_size=32)
df = pd.read_pickle('data/data_11x11.pkl')
labels = df['real'].to_numpy(dtype=int)
pt = df.pt.to_numpy()
eta = df.eta.to_numpy()
other = np.column_stack([pt, eta])
weights = np.load('data/weights_real.npy')

Handler = RechitHandler(large_file, other, labels, weights, batch_size=4096, image_size=11)
generator = Handler
print('starting batch iteration')
start = time()
for batch in generator:
    print(batch)
    # pass
end = time()
print('time', end-start)









