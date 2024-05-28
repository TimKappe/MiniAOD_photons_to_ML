import numpy as np
import pandas as pd

from numpy.typing import NDArray
from mytypes import Filename, Sparse

# rechit functions
def dense_to_sparse(dense: NDArray) -> Sparse:
    """convert dense rechit array to sparse representation
    the returned arrays are the values and the indices of the photon, row and column of the values in that order"""
    idxs_ = np.nonzero(dense)  # tuple of three arrays
    values_ = dense[idxs_]
    idxs_ = [idx_array.astype(np.int32) for idx_array in idxs_]
    return values_, *idxs_

def save_sparse_rechits(savename: Filename, sparse: Sparse) -> None:
    np.savez(savename, values=sparse[0], idx1=sparse[1], idx2 = sparse[2], idx3=sparse[3])
    print(f'INFO: file saved as {savename}')

def load_sparse(file: Filename) -> Sparse:
    npz = np.load(file)
    sparse = (npz['values'], npz['idxs1'], npz['idxs2'], npz['idxs3'])
    return sparse

def sparse_to_dense(sparse: Sparse, shape=None) -> NDArray:
    """shape can be 2d or 3d, 
    if 2d it must be the shape of the images and the number of photons will be inferred"""
    values, indices = sparse[0], sparse[1:]
    if len(shape)==2:
        num = np.max(indices[0])+1
        shape = (num, *shape)
    
    dense = np.zeros(shape, dtype=np.float32)
    dense[indices] = values
    return dense

