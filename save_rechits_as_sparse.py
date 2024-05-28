import numpy as np
import argparse

from myfunctions import dense_to_sparse, save_sparse_rechits

from typing import Tuple
from mytypes import Filename

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

sparse = dense_to_sparse(rechits)
save_sparse_rechits(outname, sparse)

print('FINISHED')

