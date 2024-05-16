import numpy as np

from typing import Generator, Tuple
from mytypes import Filename, NDArray


class RechitHandler:
    def __init__(self, file: Filename, batch_size: int, image_size: int) -> None:
        self.batch_size = batch_size
        self.image_size = image_size
        self.batch_shape = (batch_size, image_size, image_size)

        # TODO use memorymap to make this more efficient
        sparse_rechits = np.load(file)  # this is a dict with 4 entries
        self.values = sparse_rechits['values']
        self.idx_photon, self.idx_row, self.idx_col = tuple(sparse_rechits.values())[1:]
        # idx_photon indexes which photon the pixel belongs to
        # idx_row and idx_col are the idxs of the pixel in the image

        self.num_photons = self.idx_photon[-1] + 1
        self.num_batches = self.num_photons//batch_size + 1
        self.slices = self.get_slices()

        # TODO shuffle data
    
    def shuffle_data(self):
        pass
    
    def get_slices(self):  
        photon_idxs = np.unique(self.idx_photon, return_index=True)[1]
        # photon_idxs are the indices of the first appearance of each number in idxs_photons
        slices = [slice(photon_idxs[i*self.batch_size], photon_idxs[(i+1)*self.batch_size]) for i in range(self.num_batches-1)]
        slices += [slice(slices[-1].stop, len(self.idx_photon))]  # last batch is smaller if shape does not match by chance
        return slices


    def sparse_to_dense(self, values: NDArray, indices: Tuple[NDArray, NDArray, NDArray]) -> NDArray:
        dense = np.zeros(self.batch_shape, dtype=np.float32)
        dense[indices] = values
        return dense

    def batch_generator(self) -> Generator[NDArray, None, None]:
        for i, this_slice in enumerate(self.slices):
            values = self.values[this_slice]
            shifted_photon_index = self.idx_photon[this_slice] - i*self.batch_size # make sure the first index of the first array is always 0
            indices = (shifted_photon_index, self.idx_row[this_slice], self.idx_col[this_slice])
            yield self.sparse_to_dense(values, indices)


from time import time
small_file = '/home/home1/institut_3a/kappe/work/data/test_rechit_format/original_sparse.npz'
large_file = 'data/rechits_11x11_sparse.npz'
# Handler = RechitHandler(small_file, batch_size=4096, image_size=32)
Handler = RechitHandler(large_file, batch_size=4096, image_size=11)
generator = Handler.batch_generator
print('starting batch iteration')
start = time()
for batch in generator():
    # print(batch)
    pass
end = time()
print('time', end-start)









