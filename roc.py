from plotter import Scores, Data, ROC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_pickle('data/data_32x32_high.pkl')
dataframefile = 'data/data_32x32_high.pkl'
modelfiles = ['models/cnn8_pred.npy']
modelnames = ['CNN']
# modelfiles = ['models/cnn8_pred.npy',
            #   'models/cnn1_pred.npy']
# modelnames = ['CNN', 'cnn1']
weightsfile = 'data/weights_32x32_high.npy'


x = Scores(np.arange(10), name='test')

data = Data(dataframefile, modelfiles, modelnames, weightsfile, base='bdt', use_set='test')
plotter = ROC(data)

bin_edges = dict(
    pt = [25, 40, 55, 70, 100, 200],
    eta = [-1.5, -1, -0.4, 0.4, 1, 1.5],
    # eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5],  # outer barrel only
    r9 = [0.5, 0.7, 0.85, 0.9, 0.95, 1.0, 1.1],  # no r9 < 0.5 in my data
    phi = [i*60 *np.pi/180 - np.pi for i in range(int(360/60+1))],  # 60 degree steps in rad from -pi to pi
)
keys = ['pt', 'eta', 'r9', 'phi']
for key in keys:
    fig1, ax1 = plt.subplots(figsize=(10,8))
    fig2, ax2 = plt.subplots(figsize=(10,8))
    fig3, ax3 = plt.subplots(figsize=(10,8))
    fig4, ax4 = plt.subplots(figsize=(14,10))
    plotter.plot_cuts(ax1, key, bin_edges[key], mode='normal')
    plotter.plot_cuts(ax2, key, bin_edges[key], mode='ratio')
    plotter.plot_cuts(ax3, key, bin_edges[key], threshold=0.0, mode='thresholds')
    plotter.plot_cuts(ax4, key, bin_edges[key], threshold=0.0, mode='output')

    fig1name = f'plots/roc_cuts_{key}.png'
    fig2name = f'plots/roc_cuts_{key}_ratio.png'
    fig3name = f'plots/roc_cuts_{key}_thresholds.png'
    fig4name = f'plots/roc_cuts_{key}_output.png'
    fig1.savefig(fig1name)
    fig2.savefig(fig2name)
    fig3.savefig(fig3name)
    fig4.savefig(fig4name)
    print('INFO: fig saved as:', fig1name)
    print('INFO: fig saved as:', fig2name)
    print('INFO: fig saved as:', fig3name)
    print('INFO: fig saved as:', fig4name)






