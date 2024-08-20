import numpy as np
import pandas as pd

from myfunctions import load_sparse, save_sparse_rechits, combine_sparse
from myfunctions import slice_sparse, create_slice_arr, reset_sparse

from typing import List


names = 'high low mgg'.split()
dataframefiles = [f'data/data_32x32_{name}.pkl' for name in names]
rechitfiles = [f'data/rechits_32x32_{name}.npz' for name in names]
dataframes = [pd.read_pickle(file) for file in dataframefiles]
sparse_rechits = [load_sparse(file) for file in rechitfiles]

# combine dataframes
data_all = []
for df in dataframes:
    data_all += list(df.values)
labels = dataframes[0].columns.to_list()
df = pd.DataFrame(data_all, columns=labels)

# combine rechits
rechits = combine_sparse(sparse_rechits)


# shuffle dataframe and rechits together
slices = create_slice_arr(rechits)
idxs = np.arange(len(df))
np.random.shuffle(idxs)

df_shuffled = df.iloc[idxs]
rechits_shuffled = slice_sparse(rechits, idxs, slices)
rechits_shuffled = reset_sparse(rechits_shuffled)  # reorder indices

# save output
savename_df = 'data/data_shuffled.pkl'
savename_rechits = 'data/rechits_shuffled.npz'
df_shuffled.to_pickle(savename_df)
print(f'INFO: saved df as {savename_df}')
save_sparse_rechits(savename_rechits, rechits_shuffled)
print(f'INFO: saved rechits as {savename_rechits}')




