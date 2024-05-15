import numpy as np
import pandas as pd
import boost_histogram as bh
import pickle
import argparse

from typing import Tuple
Filename = str



def process_args() -> Tuple[pd.DataFrame, bh.Histogram]:
    parser = argparse.ArgumentParser(description='get weights and save them')
    parser.add_argument('--datafile', default='data.pkl', help='data to be used')
    parser.add_argument('--histfile', default='hist_weights.pkl', help='historgram of weights')
    args = parser.parse_args()

    datafile_: Filename = args.datafile
    histfile_: Filename = args.histfile

    df_ = pd.read_pickle(datafile_)

    with open(histfile_, 'rb') as f:
        weight_hist_ = pickle.load(f)

    return df_, weight_hist_

### load data
df, weight_hist = process_args()

real = df.real
fake = ~real
pt = df.pt
eta = df.eta


### decide bins:
# you could theoretically change them from the bins in the weight hist
pt_bins = np.array([0, *np.arange(25, 55, step=2.5), 55, 65, 75, 100, 125, 150, 200, 250])

num_merge = 3
eta_bins_barrel = np.linspace(-1.44, 1.44, 4*num_merge+1)  # make barrel bins smaller and merge the ones for high pt
eta_bins_endcap1 = np.linspace(-2.5, -1.57, 2)
eta_bins_endcap2 = np.linspace(1.57, 2.5, 2)
eta_bins = np.append(eta_bins_endcap1, eta_bins_barrel)
eta_bins = np.append(eta_bins, eta_bins_endcap2)  # no doublecounting because of transtion region


### get weights
pt_idxs = np.digitize(pt, pt_bins)  # digittize checks bin[i-1]<=x<bin[i] -> fits if flow=True
eta_idxs = np.digitize(eta, eta_bins)
weights = weight_hist.view(flow=True)[pt_idxs, eta_idxs]

weights_fake = weights
weights_real = 1/weights

weights_fake[real] = 1.
weights_real[fake] = 1.

weights_fake[pt > 250] = 0.
weights_real[pt > 250] = 0.


### save weights
fake_file = 'weights_fake.npy'
real_file = 'weights_real.npy'
np.save(fake_file, weights_fake)
print(f'INFO: fake weights saved as {fake_file}')
np.save(real_file, weights_real)
print(f'INFO: real weights saved as {real_file}')


print('FINISHED')
