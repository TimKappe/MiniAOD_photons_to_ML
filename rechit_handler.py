import numpy as np
from keras.utils import Sequence

from typing import Generator, Tuple, List
from mytypes import Filename, NDArray, Sparse

from mynetworks import resize_images

class RechitHandler(Sequence):
    def __init__(self, file: Filename, other_inputs: NDArray, labels: NDArray, weights: NDArray, 
                 batch_size: int, image_size: int, which_set: str = 'train') -> None:
        """set needs to be 'train', 'val' or 'test'"""
        self.which_set = which_set  #TODO remove line after debugging
        self.batch_size = batch_size
        self.image_size = image_size
        self.batch_shape = (batch_size,  image_size, image_size)

        self.other_inputs = other_inputs  # 2d array
        self.labels = labels  # 1d array
        self.weights = weights

        sparse_rechits = np.load(file)  # this is a dict with 4 entries
        self.values = sparse_rechits['values']
        self.idx_photon, self.idx_row, self.idx_col = tuple(sparse_rechits.values())[1:]
        # idx_photon indexes which photon the pixel belongs to
        # idx_row and idx_col are the idxs of the pixel in the image

        self._select_set(which_set)  # select train/val/test set only

        self.num_photons = self.idx_photon[-1] + 1
        self.num_batches = self.num_photons//batch_size + 1

        self.photon_slices, self.data_slices = self._get_slices()

        # TODO shuffle data
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, index):
        # rechits 
        this_slice = self.photon_slices[index]
        values = self.values[this_slice]
        shifted_photon_index = self.idx_photon[this_slice] - index*self.batch_size # make sure the first index of the first array is always 0
        indices = (shifted_photon_index, self.idx_row[this_slice], self.idx_col[this_slice])
        dense_rechits = self.sparse_to_dense((values, *indices), (self.image_size, self.image_size))
        dense_rechits = resize_images(dense_rechits)

        # other inputs, labels and weights
        batch_slice = self.data_slices[index]
        batched_inputs = self.other_inputs[batch_slice, :]
        batched_labels = self.labels[batch_slice]
        batched_weigths = self.weights[batch_slice]
        return [dense_rechits, batched_inputs], batched_labels, batched_weigths

    def _select_set(self, which_set: str):
        # if which_set=='train': use_data = slice(0, int(0.6*len(self.labels)))
        # elif which_set=='val': use_data = slice(int(0.6*len(self.labels)), int(0.8*len(self.labels)))
        # elif which_set=='test': use_data = slice(int(0.8*len(self.labels)), None)
        num_photons_all_sets = self.idx_photon[-1]+1
        if which_set=='train': 
            use_data = slice(0, int(0.75*len(self.labels)))
            use_photons = (self.idx_photon < int(0.6*num_photons_all_sets))
        elif which_set=='val': 
            use_data = slice(int(0.75*len(self.labels)), None)
            use_photons = (int(0.6*num_photons_all_sets) <= self.idx_photon) & (self.idx_photon < int(0.8*num_photons_all_sets))
        elif which_set=='test': 
            use_data = slice(0, None)
            use_photons = (int(0.8*num_photons_all_sets) <= self.idx_photon)
        elif which_set=='all':  
            use_data = slice(0, None)  # all data
            use_photons = (-1 <= self.idx_photon)  # all photons
        else: raise ValueError('set needs to be "train", "val" or "test')

        # TODO the error is here!!!
        # TODO it is the rounding in the different slicing operations

        # if which_set=='train': use_data = slice(0, int(0.6*num_photons_all_sets))
        # elif which_set=='val': use_data = slice(int(0.6*num_photons_all_sets), int(0.8*num_photons_all_sets))
        # elif which_set=='test': use_data = slice(int(0.8*num_photons_all_sets), None)
        # print('\n\n')
        self.values = self.values[use_photons]
        self.idx_photon = self.idx_photon[use_photons]
        self.idx_photon -= self.idx_photon[0]  # shift indices, so the first photon is always Zero
        self.idx_row = self.idx_row[use_photons]
        self.idx_col = self.idx_col[use_photons]

        # other inputs were split into train/test earlier -> need only split train into train/val
        self.other_inputs = self.other_inputs[use_data]  # 2d array
        self.labels = self.labels[use_data]  # 1d array
        self.weights = self.weights[use_data]

    
    def shuffle_data(self):
        pass
    
    def _get_slices(self) -> Tuple[List[slice], List[slice]]:  
        photon_idxs = np.unique(self.idx_photon, return_index=True)[1]
        # photon_idxs are the indices of the first appearance of each number in idxs_photons
        photon_slices = [slice(photon_idxs[i*self.batch_size], photon_idxs[(i+1)*self.batch_size]) for i in range(self.num_batches-1)]
        photon_slices += [slice(photon_slices[-1].stop, None)]  # last batch is usually smaller

        data_slices = [slice(i*self.batch_size, (i+1)*self.batch_size) for i in range(self.num_batches-1)]
        data_slices += [slice(data_slices[-1].stop, None)]  # last batch is usually smaller
        return photon_slices, data_slices

    def sparse_to_dense(self, sparse: Sparse, shape: Tuple[int, int]) -> NDArray:
        """shape can be 2d or 3d, 
        if 2d it must be the shape of the images and the number of photons will be inferred"""
        values, indices = sparse[0], sparse[1:]
        if len(shape)==2:
            num = np.max(indices[0])+1
            shape = (num, *shape)
        
        dense = np.zeros(shape, dtype=np.float32)
        dense[indices] = values
        return dense

    def get_sparse_rechits(self) -> Sparse:
        return (self.values, self.idx_photon, self.idx_row, self.idx_col)


if __name__=='__main__':
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









