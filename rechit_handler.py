import numpy as np

from typing import Generator, Tuple
from mytypes import Filename, NDArray, Mask

# rechits = np.load('data/new_rechits_pre_barrel.npy')

# indices = np.nonzero(rechits)
# values = rechits[indices]

# np.savez('data/test.npz', values=values, idx1=indices[0], idx2=indices[1], idx3=indices[2])
# print('saved_together')

# data = np.load('data/together.npz')  # this is a dict with 4 arrays
# values, indices = data['values'], tuple(data.values())[1:]
# # values_re, nonzero_re = data['values'], data


class RechitHandler:
    def __init__(self, file: Filename, batch_size: int, image_size: int) -> None:
        self.batch_size = batch_size
        self.image_size = image_size
        self.batch_shape = (batch_size, image_size, image_size)

        # TODO use memorymap to make this more efficient
        sparse_rechits = np.load(file)  # this is a dict with 4 entries
        self.values = sparse_rechits['values']
        self.idx_photon, self.idx_row, self.idx_col = tuple(sparse_rechits.values())[1:]
        # idx0 indexes which photon the pixel belongs to
        # idx1 and 2 are the idxs of the nonzero pixel in the image

        self.num_photons = self.idx_photon[-1] + 1
        self.num_batches = self.num_photons//batch_size
        self.slices = self.get_slices()

        # TODO shuffle data
    
    def shuffle_data(self):
        pass
    
    def get_slices(self):
        photon_idxs = np.unique(self.idx_photon, return_index=True)[1]
        # photon_idxs index of first appearance of each number in idxs_photons
        print(len(photon_idxs))
        print((self.num_batches-1)*self.batch_size)
        print((self.num_batches)*self.batch_size)
        slices = [slice(photon_idxs[i*self.batch_size], photon_idxs[(i+1)*self.batch_size]) for i in range(self.num_batches-1)]
        slices += [slice(slices[-1].stop, len(self.idx_photon))]  # last smaller batch
        return slices


    def sparse_to_dense(self, values: NDArray, indices: Tuple[NDArray, NDArray, NDArray]) -> NDArray:
        dense = np.zeros(self.batch_shape, dtype=np.float32)
        dense[indices] = values
        return dense

    def batch_generator(self) -> Generator[NDArray, None, None]:
        for i in range(self.num_batches):
            this_slice = self.slices[i]
            values = self.values[this_slice]
            shifted_photon_index = self.idx_photon[this_slice] - i*self.batch_size # make sure the first index of the first array is always 0
            print(this_slice)
            print(this_slice.stop - this_slice.start)
            print(shifted_photon_index[-1])
            print(i)
            print(self.num_batches)

            indices = (shifted_photon_index, self.idx_row[this_slice], self.idx_col[this_slice])
            yield self.sparse_to_dense(values, indices)


    # def old_batch_generator(self, batch_size) -> Generator[NDArray, None, None]:
    #     batch_shape = (batch_size, self.image_size, self.image_size)

    #     num_photons = self.idx0[-1] + 2  # +2 because both range and my slicing is exclusive and I need to reach the last values as well
    #     start_photon = 0
    #     for end_photon in range(batch_size, num_photons, batch_size):  # end_photon starts at batchsize, because start_photon starts at 0
    #         # print('\n\ncurrent end photon:', end_photon)
    #         this_batch: Mask = (start_photon <= self.idx0) & (self.idx0 < end_photon)
    #         print(end_photon)
    #         photon_idx_in_batch = self.idx0[this_batch] - start_photon
    #         image_idxs_in_batch = (self.idx1[this_batch], self.idx2[this_batch])
    #         batch_values = self.values[this_batch]
    #         batch_idxs = (photon_idx_in_batch, *image_idxs_in_batch)
            
    #         batch = self.sparse_to_dense(batch_values, batch_idxs, batch_shape)

    #         start_photon = end_photon
    #         yield batch

Handler = RechitHandler('data/test.npz', batch_size=5, image_size=11)
generator = Handler.batch_generator
for batch in generator():
    print(batch)











