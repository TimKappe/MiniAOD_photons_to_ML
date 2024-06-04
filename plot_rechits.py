import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import boost_histogram as bh
import matplotlib.colors as colors
import copy

import myplotparams
from myfunctions import load_sparse, sparse_to_dense


def plot_image(image, title, savename=None):
    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under('w')

    image[image<1e-6]=1e-6

    plt.figure()
    im = plt.imshow(image, norm=colors.LogNorm(vmin=1e-6, vmax=image.max()), cmap=cmap, interpolation=None)
    plt.colorbar(im, label='Energy deposition [GeV]')
    plt.xlabel("iphi")
    plt.ylabel("ieta")
    plt.title(title)
    if savename is not None:
        plt.savefig(savename)
        print(f'INFO: plot saved as {savename}')
    # plt.clf()

def load_rechits(file='combined.npy'):
    return np.load(file)


df = pd.read_pickle("data/data_32x32_high.pkl")
sparse = load_sparse('data/rechits_32x32_high.npz')

dense = sparse_to_dense(sparse, (32, 32))


pt = df.pt.to_numpy()
real = df.real
fake = ~real

idxs = np.arange(len(real))
random_real = np.random.randint(0, real.sum(), 10)
random_fake = np.random.randint(0, fake.sum(), 10)
idxs_real = idxs[real][random_real]
idxs_fake = idxs[fake][random_fake]
for i in range(len(random_fake)):
    print(i+1)
    idx = idxs_real[i]
    # print(dense[real[idx]])
    plot_image(dense[idx], f'real, pt={pt[idx]:.2f}', savename=f'plots/image_real{i+1}.png')
    idx = idxs_fake[i]
    plot_image(dense[idx], f'fake, pt={pt[idx]:.2f}', savename=f'plots/image_fake{i+1}.png')


plt.show()


