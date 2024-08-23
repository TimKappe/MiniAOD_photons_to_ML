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
from mynetworks import load_and_prepare_data
from myparameters import Parameters, data_from_params, weights_from_params
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

def plot_roc_classic(axis: plt.Axes, predictions: NDArray, true_values: NDArray, weights: NDArray,
             threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
    """plots ROC as usual in HEP = true positive rate vs background rejection (1/fpr), xlim will be > threshold
    if threshold is undesiered, set to any negative value or None"""
    if threshold is None:
        threshold: float = -1.
    fpr, tpr, _ = skm.roc_curve(true_values, predictions, sample_weight=weights)
    mask: Mask = tpr > threshold

    axis.plot(tpr[mask], fpr[mask], linewidth=2, **kwargs)
    if fifty_percent_line:
        fifty: float = get_tpr(0.5, predictions, true_values)
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)

    axis.set_title('ROC classic')
    axis.set_xlabel('True positive rate')
    axis.set_ylabel('False positive rate')
    axis.set_xlim(threshold, 1.0)

    axis.grid(True)
    axis.legend(loc='upper left')
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


def process_parser() -> Tuple[List[Parameters], Union[Filename, str], Filename]:
    parser = argparse.ArgumentParser(description='plot roc of models ', prog='plot_roc_dist.py')
    parser.add_argument('parameterfilenames', nargs='+', help='model to be used')
    parser.add_argument('--base', help='network to compare to in ratio plot. if not set, the BDT is used')
    parser.add_argument('--figname', default='models/roc.png')

    args = parser.parse_args()
    param_list_: List[Parameters] = [Parameters(load=file) for file in args.parameterfilenames]
    check_params_work_together(param_list_)

    figname_: Filename = args.figname
    base_: Optional[Union[Filename, str]] = args.base
    if base_ is None:
        base_ = 'bdt'
    print(f'\nBase is {base_}')
    return param_list_, base_, figname_


### load and prepare the data
param_list, base, figname = process_parser()

df = pd.read_pickle(param_list[0]['dataframefile'])
_, weights_test = weights_from_params(param_list[0])

_, y_test = split_data(param_list[0], df['real'].to_numpy(dtype=int))

### get the model predictions
y_pred_list = [np.load(param['modeldir'] + param['modelname'] + '_pred.npy') for param in param_list]

bdt = df['bdt3'].to_numpy()
_, pred_bdt = split_data(param_list[0], bdt)


# set/load base pred to compare to
if base.lower() == 'bdt':
    base_name = 'BDT'
    base_pred = pred_bdt
else:
    base_params = Parameters(load=base)
    base_name = base_params['modelname']
    base_pred = np.load(base_params['modeldir'] + base_params['modelname'] + '_pred.npy')

######################################################################################
### plot ROC
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
plot_roc(ax1, pred_bdt, y_test, weights_test, label='BDT', color='grey')#, color='orange')
plot_roc(ax1, base_pred, y_test, weights_test, label=base_name, color='black')
plot_roc_classic(ax3, pred_bdt, y_test, weights_test, label='BDT', color='grey')#, color='orange')
plot_roc_classic(ax3, base_pred, y_test, weights_test, label=base_name, color='black')

for i, param in enumerate(param_list):
    plot_roc(ax1, y_pred_list[i], y_test, weights_test, label=param['modelname'])
    plot_roc_classic(ax3, y_pred_list[i], y_test, weights_test, label=param['modelname'])
    plot_roc_ratio(ax2, base_pred, y_pred_list[i], y_test, weights_test, label=f'{param["modelname"]}/{base_name}')
# ax2.set_title(None)
# ax2.set_ylabel('ratio')
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
# plt.subplots_adjust(hspace=0.0)

if figname is not None: 
    fig2name = f'{figname.split(".")[0]}_ratio.png'
    fig3name = f'{figname.split(".")[0]}_classic.png'
    fig1.savefig(figname)
    fig2.savefig(fig2name)
    # fig3.savefig(fig3name)  # don't usually need classic anymore
    print('INFO: fig saves as:', figname, fig2name)#, fig3name)

modelfiles = [
            #   'models/cnn_only_image_pred.npy', 
              'models/cnn0_pred.npy', 
            #   'models/cnn1_pred.npy', 
            #   'models/cnn2_pred.npy', 
              'models/cnn3_pred.npy', 
            #   'models/cnn4_pred.npy', 
              'models/cnn5_pred.npy', 
              'models/cnn6_pred.npy', 
              'models/cnn7_pred.npy', 
              'models/cnn8_pred.npy', 
              ]
modelnames = [
            #   'no additional inputs',
              r'$ p_t $', 
            #   r'$ \eta $', 
            #   r'$ \rho $', 
              r'$ \eta, \rho, \varphi $', 
            #   r'$ \frac{H}{E}, hcalIso, I_{tr}, I_{ch}, I_n $', 
            #   r'$ ecalIso, I_{\gamma} $', 
              r'$ \frac{H}{E}, hcalIso, I_{tr}, I_{ch}, I_n $  \n  $ ecalIso, I_{\gamma} $', 
              r'$ \sigma_{i\eta i\eta} $', 
              r'$ R_9 $', 
              r'conversion info', 
              ]
fignames = [f'models/roc_inputs_{i}.png' for i in range(len(modelnames))]

fig1, ax1 = plt.subplots(figsize=(10,8))
fig2, ax2 = plt.subplots(figsize=(14,8))  # larger because of outside legend
plot_roc(ax1, pred_bdt, y_test, weights_test, label='BDT', color='black', ls='--')
color=None
for i in range(len(modelfiles)):
    pred = np.load(modelfiles[i])
    if modelfiles[i]==modelfiles[-1]:
        color='black'
    plot_roc(ax1, pred, y_test, weights_test, label=modelnames[i], color=color)
    plot_roc_ratio(ax2, pred_bdt, pred, y_test, weights_test, label=modelnames[i], color=color)
    
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax2.set_ylim(None, 3.8)
    
    ax2.set_title('ROC ratios: CNN/BDT')

    fig1.tight_layout()	
    fig2.tight_layout()	

    name = fignames[i].split('.')[0] + '_ratio.png'
    fig1.savefig(fignames[i])
    fig2.savefig(name)
    print('INFO: fig saved as:', fignames[i])
    print('INFO: fig saved as:', name)



print('FINISHED')
plt.show()

