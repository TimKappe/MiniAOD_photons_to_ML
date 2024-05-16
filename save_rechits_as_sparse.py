import numpy as np
import argparse

from typing import Tuple
from mytypes import Filename, NDArray


def process_args() -> Tuple[Filename, Filename]:
    parser = argparse.ArgumentParser(description='read a dense rechits file and save it in sparse representation',
                                     prog='save_rechits_as_sparse.py')
    parser.add_argument('file', help='file to be read')
    parser.add_argument('outname', help='name of the outputfile')

    args = parser.parse_args()
    file_: Filename = args.file
    outname_: Filename = args.outname
    if outname_[-4:] != '.npz':
        raise ValueError('"outname" needs to be a .npz-file')
    return file_, outname_

file, outname = process_args()
rechits = np.load(file)

nonzero = np.nonzero(rechits)  # tuple of three arrays
values = rechits[nonzero]
nonzero = [idx_array.astype(np.int32) for idx_array in nonzero]  # change from int64 to save diskspace

np.savez(outname, values=values, idx1=nonzero[0], idx2 = nonzero[1], idx3=nonzero[2])
print(f'INFO: file saved as {outname}')


print('FINISHED')

