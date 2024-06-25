import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow import keras
# import tensorflow as tf
import keras
import argparse
import sklearn.metrics as skm

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Mask, Filename

import myplotparams

# my functions
from myparameters import Parameters, weights_from_params
from myparameters import check_params_work_together
from myparameters import split_data
# shorthands
# layers = keras.layers
# models = keras.models
# tf.get_logger().setLevel('ERROR')


def get_tpr(cut: float, predictions: NDArray, true_values: NDArray,) -> float:
    larger: Mask = predictions > cut
    true_positives: Mask = true_values[larger]
    tpr: float = true_positives.sum()/true_values.sum()
    return tpr

def interpolate(x: NDArray, y: NDArray, x_new: NDArray) -> NDArray:
    # x is tpr
    # y is fpr
    return np.interp(x_new, x, y)


def plot_roc(axis: plt.Axes, predictions: NDArray, true_values: NDArray, weights: NDArray,
             threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
    """plots ROC as usual in HEP = true positive rate vs background rejection (1/fpr), xlim will be > threshold
    if threshold is undesiered, set to any negative value or None"""
    if threshold is None:
        threshold: float = -1.
    fpr, tpr, _ = skm.roc_curve(true_values, predictions, sample_weight=weights)
    mask: Mask = tpr > threshold
    axis.plot(tpr[mask], 1/(fpr[mask]), linewidth=2, **kwargs)
    if fifty_percent_line:
        fifty: float = get_tpr(0.5, predictions, true_values)
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)

    axis.set_title('ROC')
    axis.set_xlabel('True positives rate')
    axis.set_ylabel('Background rejection')
    axis.set_xlim(threshold, 1.0)
    # axis.set_ylim(0, 1)
    axis.grid(True)
    axis.grid(True)
    axis.grid(True)
    # axis.set_aspect('equal')
    axis.legend(loc='upper right')
    plt.tight_layout()

def plot_roc_ratio(axis: plt.Axes, predictions_base: NDArray, predictions_compare: NDArray, true_values: NDArray,
                   weights: NDArray, 
                   threshold: float = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:

    fpr_base, tpr_base, _ = skm.roc_curve(true_values, predictions_base, sample_weight=weights)
    fpr_comp, tpr_comp, _ = skm.roc_curve(true_values, predictions_compare, sample_weight=weights)
    mask: Mask = tpr_base > threshold

    fpr_interp = np.interp(tpr_base, tpr_comp, fpr_comp)

    rej_base = 1/fpr_base
    rej_interp = 1/fpr_interp

    axis.plot(tpr_base[mask], rej_interp[mask]/rej_base[mask], linewidth=2, **kwargs)

    if fifty_percent_line:
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        fifty: float = get_tpr(0.5, predictions_compare, true_values)
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)
    axis.axhline(1, color='black', alpha=0.8, ls='--', zorder=-1)

    axis.set_title('ROC ratios')
    axis.set_xlabel('True positives rate')
    axis.set_ylabel('Background rejection ratio')
    axis.set_xlim(threshold, 1.)
    # ylim = max(np.abs(1-np.array(axis.get_ylim())))
    # axis.set_ylim(1-ylim, 1+ylim)

    axis.grid(True)
    axis.legend(loc='upper left')
    plt.tight_layout()


def process_parser() -> Tuple[Parameters, Union[Filename, str], Filename]:
    parser = argparse.ArgumentParser(description='plot roc of models ', prog='plot_roc_dist.py')
    parser.add_argument('parameterfile', help='model to be used')
    parser.add_argument('--base', help='network to compare to in ratio plot. if not set, the BDT is used')
    parser.add_argument('--figname', default='models/roc.png')

    args = parser.parse_args()
    param_: Parameters = Parameters(load=args.parameterfile)

    figname_: Filename = args.figname
    base_: Optional[Union[Filename, str]] = args.base
    if base_ is None:
        base_ = 'bdt'
    print(f'\nBase is {base_}')
    return param_, base_, figname_


### load and prepare the data
params, base, figname = process_parser()

df = pd.read_pickle(params['dataframefile'])
_, weights_test = weights_from_params(params)

_, y_test = split_data(params, df['real'].to_numpy(dtype=int))

### get the model predictions
y_pred = np.load(params['modeldir'] + params['modelname'] + '_pred.npy')

bdt = df['bdt3'].to_numpy()
_, pred_bdt = split_data(params, bdt)

print('\n\n\n')
pred_bdt = pred_bdt*2 -1  # convert BDT back to what it was in the MiniAOD
# pred_bdt = 0.5 * np.log((2.0 / (1.0 - pred_bdt) - 1.0))
print((y_pred>0.5).sum())
y_pred = 1.0 - 2.0 / (1.0 + np.exp(2.0 * y_pred))  # convert my classifier to -1 to 1 in a weird way
y_test[y_test==0] = -1
print((y_pred>0).sum())
print()
print((y_pred>0.5).sum())
print('FUCK OFFFFFFFFFFFF')
print('\n\n\n')

# set/load base pred to compare to
if base.lower() == 'bdt':
    base_name = 'BDT'
    base_pred = pred_bdt
else:
    base_params = Parameters(load=base)
    base_name = base_params['modelname']
    base_pred = np.load(base_params['modeldir'] + base_params['modelname'] + '_pred.npy')

pt_bin_edges = [25, 40, 55, 70, 100, 200]
pt_bins = list(zip(pt_bin_edges[:-1], pt_bin_edges[1:]))
_, pt = split_data(params, df.pt)
######################################################################################
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # standard matplotlib color cycle

line_color = lambda i: plt.Line2D([0], [0], color=colors[i], 
                            label=f'${pt_bins[i][0]} \leq p_t < {pt_bins[i][1]}$') 
color_legend_elements = [line_color(i) for i in range(len(pt_bins))]
color_legend_elements += [plt.Line2D([0], [0], color='black', label=r'full $p_t$ range') ]

linestyle_legend_elements = [
    plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label=params['modelname']),
    plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label=base_name) # most likey BDT,
    ]
# Create the first legend for colors

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

for i, (min_pt, max_pt) in enumerate(pt_bins):
    mask = (min_pt <= pt) & (pt < max_pt)
    current_color = colors[i]

    # plot_roc(ax1, pred_bdt[mask], y_test[mask], weights_test[mask], label='BDT', color='grey')
    # base
    plot_roc(ax1, base_pred[mask], y_test[mask], weights_test[mask], threshold=0.6, label=base_name, ls='--', color=current_color)

    plot_roc(ax1, y_pred[mask], y_test[mask], weights_test[mask], threshold=0.6, label=params['modelname'], color=current_color)
    plot_roc_ratio(ax2, base_pred[mask], y_pred[mask], y_test[mask], weights_test[mask], threshold=0.6, label=f'{params["modelname"]}/{base_name}')

# plot full pt range for comparison
plot_roc(ax1, base_pred, y_test, weights_test, threshold=0.6, ls='--', color='black')  # labels set in color_legend_elements
plot_roc(ax1, y_pred, y_test, weights_test, threshold=0.6, color='black')  # labels set in color_legend_elements
plot_roc_ratio(ax2, base_pred, y_pred, y_test, weights_test, threshold=0.6, color='black')

# add legends
first_legend = ax1.legend(handles=color_legend_elements, loc='upper right')
ax1.add_artist(first_legend)
ax1.legend(handles=linestyle_legend_elements, loc='center right')

ax2.legend(handles=color_legend_elements, loc='upper right')


fig1.tight_layout()
fig2.tight_layout()

if figname is not None: 
    fig1name = f'{figname.split(".")[0]}.png'
    fig2name = f'{figname.split(".")[0]}_ratio.png'
    fig1.savefig(fig1name)
    fig2.savefig(fig2name)
    print('INFO: fig saves as:', fig1name, fig2name)


print('FINISHED')

