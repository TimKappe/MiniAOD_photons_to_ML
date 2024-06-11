import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import boost_histogram as bh
import matplotlib.colors as colors
import copy

from myfunctions import load_sparse, sparse_to_dense
import myplotparams

from typing import Tuple, Optional
from mytypes import Mask, NDArray, Sparse, Filename

def plot_image(image, title=None, savename=None):
    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under('w')

    image[image<1e-6]=1e-6

    plt.figure()
    im = plt.imshow(image, norm=colors.LogNorm(vmin=1e-6, vmax=image.max()), cmap=cmap, interpolation=None)
    plt.colorbar(im, label='Energy deposition [GeV]')
    plt.xlabel("iphi")
    plt.ylabel("ieta")
    plt.title(title)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
        print(f'INFO: plot saved as {savename}')
    # plt.clf()

def plot_average_rechits(rechits: NDArray, title: Optional[str] = None, savename: Optional[Filename] = None) -> None:
    num = len(rechits)
    average = rechits.mean(axis=(0))
    plot_image(average, title, savename)




df = pd.read_pickle("data/data_32x32_high.pkl")
sparse = load_sparse('data/rechits_32x32_high.npz')

print('data loaded')
dense = sparse_to_dense(sparse, (32, 32))
print('rechits made dense')

pt = df.pt.to_numpy()
real = df.real
fake = ~real
converted = df.converted
convertedOneLeg = df.convertedOneLeg

both = converted & convertedOneLeg
not_converted = ~(converted | convertedOneLeg)

combinations = [not_converted, converted, convertedOneLeg, both]
names = ['not converted', 'converted', 'oneLeg', 'both']
for sel, realfake in zip([real, fake], ['real', 'fake']):
    for mask, name in zip(combinations, names):
        savename = f'plots/average_{realfake}_{name}.png'
        title = f'{realfake}, {name}'
        plot_average_rechits(dense[mask & sel], title, savename)




# num = 50
# idxs = np.arange(len(df))
# random_real = np.random.randint(0, real.sum(), num)
# random_fake = np.random.randint(0, fake.sum(), num)
# idxs_real = idxs[real][random_real]
# idxs_fake = idxs[fake][random_fake]
# print('i: index, converted, convertedOneLeg')
# for i in range(len(random_fake)):
#     idx = idxs_real[i]
#     c, col = converted[real][idx], convertedOneLeg[real][idx]
#     print(i+1, ':', idx, c, col)
#     # print(dense[real[idx]])
#     plot_image(dense[idx], f'real, pt={pt[idx]:.2f}', savename=f'plots/image_real{i+1:02d}.png')
#     idx = idxs_fake[i]
#     plot_image(dense[idx], f'fake, pt={pt[idx]:.2f}', savename=f'plots/image_fake{i+1:02d}.png')


plt.show()


