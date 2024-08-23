import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import boost_histogram as bh
# from tensorflow import keras
# import tensorflow as tf
import keras
import argparse
import sklearn.metrics as skm

from typing import List, Tuple, Optional, Union, Any
from typing import Iterable
from numpy.typing import NDArray
from mytypes import Mask, Filename



class NamedArray(np.ndarray):
    """this is basically an array a name attribute. it can also be loaded from a .npy-file"""
    def __new__(cls, input_array: Union[NDArray, Filename], name: str = ""):
        # Create the ndarray instance
        if isinstance(input_array, str):
            input_array = np.load(input_array)
        obj = np.asarray(input_array).view(cls)
        # Add the new attribute
        obj.name = name
        return obj

    def __array_finalize__(self, obj: Union[np.ndarray, 'NamedArray']) -> None:
        if obj is None: return
        # Ensure that name attribute is carried over to new objects
        self.name = getattr(obj, 'name', "")

    def __reduce__(self) -> Tuple[Any, Any, Tuple[Any, ...]]:
        # Include the name attribute in the pickle process
        pickled_state = super(NamedArray, self).__reduce__()
        new_state = pickled_state[2] + (self.name,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state: Tuple[Any, ...]) -> None:
        # Restore the name attribute from the pickle
        self.name = state[-1]
        super(NamedArray, self).__setstate__(state[:-1])


class Scores(NamedArray):
    """will load itself"""
    # def __init__(self, ) -> None:
        # super.__init__()
    
    def set_name(self, new_name: str) -> None:
        self.name = new_name        

    def set_scores(self, new_scores: NDArray) -> None:
        self.scores = new_scores

# x = Scores('models/cnn8_pred.npy', name='cnn8')
# print('\n\n\n')
# print(x.name)
# x.set_name('blub')
# print(x.name)

#selct is for later
def select(self, arr: NDArray, key_or_array: Union[str, NDArray], minimum: Union[float, None], maximum: Union[float]) -> Mask:
    if isinstance(key_or_array, str):
        selector = self.df[key_or_array]
    else:
        selector = key_or_array
    sel: Mask = np.ones(arr.shape, dtype=bool)
    if minimum is not None:
        sel &= (minimum < selector)
    elif maximum is not None:
        sel &= (selector < maximum)
    else:
        return ValueError('minimum and maximum cannot both be None')
    return arr[sel]


class Data:
    """contains the df, the truth and a dict of scores"""
    def __init__(self, dataframefile: Filename, predictionfiles: List[str], modelnames: List[str], 
                 weightfile: Optional[Filename] = None, base: str = 'bdt', base_file=None, use_set: str = 'test'
                 ) -> None:
        self.df = pd.read_pickle(dataframefile)
        self.df['fake'] = ~self.df['real']
        self.truth: NDArray = self.df['real'].to_numpy(dtype=int)

        # correct incorrectly transformed bdt values:
        # pred_bdt = self.df['bdt3']
        # print(pred_bdt.min(), pred_bdt.max())
        # pred_bdt = pred_bdt*2 -1  # back to values in MINIAOD
        # print(pred_bdt.min(), pred_bdt.max())
        # # pred_bdt = 0.5 * np.log(2.0 / (1.0 - pred_bdt) - 1.0)  # correct trafo
        # # pred_bdt = 1.0 - 2.0 / (1.0 + np.exp(2.0 * pred_bdt))
        # print(pred_bdt.min(), pred_bdt.max())
        # self.df['bdt3'] = pred_bdt


        if weightfile is None:
            self.weights = np.ones(self.truth.shape, dtype=bool)
        else:
            self.weights = np.load(weightfile)

        self.base = self._set_base(base, base_file)
        self.scores = {name: Scores(file, name) for file, name in zip(predictionfiles, modelnames)}

        self._select_set(use_set)
        self.scores['bdt'] = self.df['bdt3'].to_numpy()
    
    def _set_base(self, base: str, file: Optional[Filename] = None) -> Scores:
        """returns the Scores of either the bdt or a model loaded from a parameterfilename"""
        if base.lower() == 'bdt':
            base_name = 'BDT'
            base_pred = self.df['bdt3'].to_numpy()
        elif file is not None: #TODO fix
            base_name = base
            base_pred = np.load(file)
        return Scores(base_pred, base_name)

    def split(self, array_like, mini: Union[int, float, None], maxi: Union[int, float, None]):
        """slice array_like from mini to maxi
        if mini and maxi are between 0 and 1, they are interpreted as percentages of the size
        """
        size = len(array_like)
        if (0<mini) & (mini<1) & (mini is not None):
            mini = int(mini*size)
        if (0<maxi) & (maxi<1) & (maxi is not None):
            maxi = int(maxi*size)
        return array_like[slice(mini, maxi)]

    def _select_set(self, use_set):
        sets = {'train': (None, 0.6), 'val': (0.6, 0.8), 'test': (0.8, None), 'all': (None, None)}
        num = len(self.truth)
        borders = [None, None]
        for i, fraction in enumerate(sets[use_set]):
            if fraction is None: continue
            borders[i] = int(fraction*num)
        select = slice(*borders)
        # print(select.stop - select.start)
        self.df = self.df[select]
        self.truth = self.truth[select]
        # self.scores = {key: arr[select] for (key, arr) in self.scores.items()}  
        self.base = self.base[select]
        self.weights = self.weights[select]





class ROC:
    def __init__(self, data: Data, colors: Optional[List[str]] = None, styles: Optional[List[str]] = None) -> None:
        # TODO better init
        self.data = data
        self.plotting_functions = {'normal': self.plot_roc, 
                                   'ratio': self.plot_roc_ratio, 
                                   'classic': self.plot_roc_classic,
                                   'thresholds': self.plot_thresholds,
                                   'output': self.plot_output,
                                   }
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # standard matplotlib color cycle
        self.styles = ['-', '--', ':', '-.']
        if colors is not None:
            self.colors = colors
        if styles is not None:
            self.styles = styles

    # def plot(self):
    def plot_roc(self, axis: plt.Axes, predictions: NDArray, selection: Optional[Mask] = None,
            threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
        """plots ROC as usual in HEP = true positive rate vs background rejection (1/fpr), xlim will be > threshold
        if threshold is undesiered, set to None"""
         # load stuff
        truth, weights = self._get_truth_and_weights()
        if selection is not None:
            truth, predictions, weights = self.apply_mask(selection, (truth, predictions, weights))
        
        # calc tpr, fpr and apply threshold
        tpr, fpr = self.get_tpr_fpr(truth, predictions, weights)
        tpr, fpr = self.apply_mask(tpr > threshold, (tpr, fpr))

        rej = 1/fpr
        
        # plotting
        axis.plot(tpr, rej, linewidth=2, **kwargs)
        if fifty_percent_line:
            fifty_label = kwargs.get('label') + r' 50% output'
            self.plot_fifty_percent_line(axis, predictions, truth, label=fifty_label)
        
        # labels and formatting
        axis.set_title('ROC')
        axis.set_xlabel('True positives rate')
        axis.set_ylabel('Background rejection')
        axis.set_xlim(threshold, 1.0)
        axis.grid(True)
        axis.legend(loc='upper right')
        plt.tight_layout()

    def plot_roc_classic(self, axis: plt.Axes, predictions: NDArray, selection: Optional[Mask] = None,
             threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
        """plots ROC as usual in Deep Learning = true positive rate vs false positive rate, xlim will be > threshold
        if threshold is undesired, set to None"""
         # load stuff
        truth, weights = self._get_truth_and_weights()
        if selection is not None:
            truth, predictions, weights = self.apply_mask(selection, (truth, predictions, weights))
        
        # calc tpr, fpr and apply threshold
        tpr, fpr = self.get_tpr_fpr(truth, predictions, weights)
        tpr, fpr = self.apply_mask(tpr > threshold, (tpr, fpr))

        # plotting
        axis.plot(tpr, fpr, linewidth=2, **kwargs)
        if fifty_percent_line:
            fifty_label = kwargs.get('label') + r' 50% output'
            self.plot_fifty_percent_line(axis, predictions, truth, label=fifty_label)

        # labels and formatting
        axis.set_title('ROC classic')
        axis.set_xlabel('True positive rate')
        axis.set_ylabel('False positive rate')
        axis.set_xlim(threshold, 1.0)

        axis.grid(True)
        axis.legend(loc='upper left')
        plt.tight_layout()

    def plot_roc_ratio(self, axis: plt.Axes, predictions: NDArray, selection: Optional[Mask] = None,
                    threshold: float = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
        # load stuff
        truth, weights = self._get_truth_and_weights()
        base_prediction, base_name = self.data.base, self.data.base.name
        if selection is not None:
            truth, predictions, base_prediction, weights = self.apply_mask(selection, (truth, predictions, base_prediction, weights))
        
        # calc tpr, fpr and apply threshold
        tpr_base, fpr_base = self.get_tpr_fpr(truth, base_prediction, weights)
        tpr_comp, fpr_comp = self.get_tpr_fpr(truth, predictions, weights)

        fpr_interp = np.interp(tpr_base, tpr_comp, fpr_comp)

        mask = tpr_base > threshold
        tpr_base, fpr_base, fpr_interp = self.apply_mask(mask, (tpr_base, fpr_base, fpr_interp))

        rej_base = 1/fpr_base
        rej_interp = 1/fpr_interp
        ratio = rej_interp/rej_base

        # plotting
        axis.plot(tpr_base, ratio, linewidth=2, **kwargs)
        axis.axhline(1, color='black', alpha=0.8, ls='--', zorder=-1)

        if fifty_percent_line:
            fifty_label = kwargs.get('label') + r'0.5 score'
            self.plot_fifty_percent_line(axis, predictions, truth, color=kwargs.get('color'), label=fifty_label)

        # labels and formatting
        axis.set_title('ROC ratios')
        axis.set_xlabel('True positives rate')
        axis.set_ylabel('Background rejection ratio')
        axis.set_xlim(threshold, 1.)

        axis.grid(True)
        axis.legend(loc='upper left')
        plt.tight_layout()
    
    def plot_thresholds(self, axis: plt.Axes, predictions: NDArray, selection: Optional[Mask] = None,
                threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
            """plots ROC as usual in HEP = true positive rate vs background rejection (1/fpr), xlim will be > threshold
            if threshold is undesiered, set to None"""
            # load stuff
            truth, weights = self._get_truth_and_weights()
            if selection is not None:
                truth, predictions, weights = self.apply_mask(selection, (truth, predictions, weights))
            
            # calc tpr, fpr and apply threshold
            fpr, tpr, thresholds = skm.roc_curve(truth, predictions, sample_weight=weights)
            tpr, fpr, thresholds = self.apply_mask(tpr > threshold, (tpr, fpr, thresholds))

            rej = 1/fpr
            
            # plotting
            # axis.plot(tpr, thresholds, linewidth=2, **kwargs)
            axis.plot(thresholds, tpr, linewidth=2, **kwargs)
            axis.plot(thresholds, fpr, linewidth=2, **kwargs)
            axis.set_ylim(0,1)

            if fifty_percent_line:
                fifty_label = kwargs.get('label') + r' 50% output'
                self.plot_fifty_percent_line(axis, predictions, truth, label=fifty_label)
            
            # labels and formatting
            axis.set_title('Decision Thresholds')
            # axis.set_xlabel('True positives rate')
            # axis.set_ylabel('model decision threshold')
            axis.set_xlabel('model decision threshold')
            axis.set_ylabel('tpr | fpr')
            # axis.set_xlim(threshold, 1.0)
            axis.grid(True)
            axis.legend(loc='upper right')
            # plt.tight_layout()
    
    def plot_output(self, axis: plt.Axes, predictions: NDArray, selection: Optional[Mask] = None,
                    output_bins: Union[float, int] = 20,
            threshold: Union[float, None] = 0.6, fifty_percent_line: bool = False, **kwargs) -> None:
        """fifty percent line does not do anything and is only for compytibility"""
        if output_bins < 1:
            binning = (int(1/output_bins), 0, 1)
        else: 
            binning = (int(output_bins), 0, 1)
        
        def histplot(ax: plt.Axes, data: NDArray, binning: Tuple[int, float, float],
             normalizer: Optional[Union[bool, int, float, NDArray]] = None,
             weights: Optional[NDArray] = None,
             **kwargs) -> bh.Histogram:
            """
            put data in a boosthist with binning for bh.Axes.Regular and plot it along the giving axis
            data is normalized to normalizer: int or summable, is set to 1 if False or set to hist.sum() if None
            kwargs are passed to plot
            returns the histogram
            """
            hist = bh.Histogram(bh.axis.Regular(*binning))
            hist.fill(data)
            if weights is not None:
                hist *= weights
            if normalizer is None:
                normalizer = hist.sum()
            elif normalizer is False:
                normalizer = 1
            elif isinstance(normalizer, bh.Histogram):
                normalizer = normalizer.sum()
            else:
                normalizer = bh.Histogram(bh.axis.Regular(*binning)).fill(normalizer).sum()

            values = hist.view() / normalizer
            error = np.sqrt(hist.view()) / normalizer
            ax.errorbar(hist.axes[0].centers, values, yerr=error, ds='steps-mid', **kwargs)
            return hist


         # load stuff
        _, weights = self._get_truth_and_weights()
        real = self.data.df.real
        if selection is not None:
            predictions, weights, real = self.apply_mask(selection, (predictions, weights, real))

        # plotting        
        fake = ~real
        histplot(axis, predictions[real], binning, **kwargs)
        histplot(axis, predictions[fake], binning, **kwargs)

        # labels and formatting
        axis.set_title('Output distribution')
        axis.set_xlabel('Score')
        axis.set_ylabel('#')
        axis.grid(True)
        axis.legend(loc='upper center')
        plt.tight_layout()


    def _get_truth_and_weights(self) -> Tuple[NDArray, NDArray]:
        return self.data.truth, self.data.weights

    def get_tpr_fpr(self, truth: NDArray, scores: Scores, weights: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
        if weights is None:
            weights = np.ones(truth.shape)
        fpr, tpr, thresholds = skm.roc_curve(truth, scores, sample_weight=weights)
        idx = np.argmin(np.abs(tpr-0.8))
        print('score threshold:', thresholds[idx])
        return tpr, fpr

    def interpolate(x: NDArray, y: NDArray, x_new: NDArray) -> NDArray:
        """
        the points (x, y) will get interpolated to the points (x_new, y_new)
        returns y_new
        in the case of ROCs x is the tpr and y is the fpr
        """
        return np.interp(x_new, x, y)   

    def get_tpr(self, cut: float, predictions: NDArray, true_values: NDArray,) -> float:
        """returns the tpr correpsonding to a certain cut in the predictions"""
        larger: Mask = predictions > cut
        true_positives: Mask = true_values[larger]
        tpr: float = true_positives.sum()/true_values.sum()
        return tpr

    def apply_mask(self, mask: Mask, arrays: Iterable[NDArray]) -> Tuple[NDArray, ...]:
        """applies mask to an arbitrary amount of arrays"""
        out = []
        for arr in arrays:
            out.append(arr[mask])
        return tuple(out)

    def plot_fifty_percent_line(self, axis, scores: Scores, truth: NDArray, **kwargs) -> None:
        fifty: float = self.get_tpr(0.5, scores, truth)
        label: str = '50% mark'
        if kwargs['label']: label = kwargs.get('label') + ' 50% mark'
        axis.axvline(fifty, ls='--', color=kwargs.get('color'), label=label)


    def plot_models(self, axis: plt.Axes, mode: str = 'ratio', **kwargs) -> None:
        plot_mode = self.plotting_functions[mode.lower()]
        for i, (name, score) in enumerate(self.data.scores.items()):
            current_color = self.colors[i]
            plot_mode(axis, score, label=name, color=current_color, **kwargs)

    
    def plot_cuts(self, axis: plt.Axes, key: str, bin_edges: List[int], mode: str = 'ratio', 
                  threshold: float = 0.6, fifty_percent_line: bool = False, apply_to='both', **kwargs) -> None:
        """
        mode can be "ratio", "normal" or "classic"
        apply to can be "both", "real" or "fake"
        """
        plot_mode = self.plotting_functions[mode.lower()]
        bins = list(zip(bin_edges[:-1], bin_edges[1:]))
        latex_keys = {'pt': '$p_t$',
                      'eta': '$\eta$',
                      'phi': '$\phi$',
                      'r9': '$R_9$',
                      'sigma_ieie': r'$\sigma_{ieie}$',
                      }
        # preparation
        label_func  = lambda lower, upper: f'{lower} $ \leq $ {latex_keys[key]} $ < $ {upper}'
        labels = [label_func(lower, upper) for (lower, upper) in bins]
        full_label = f'full dataset'
        colors = self.colors[:len(labels)]
        styles = self.styles

        line = lambda color, label, style: plt.Line2D([0], [0], color=color, label=label, ls=style)
        color_line = lambda color, label: line(color, label, '-')
        style_line = lambda style, label: line('black', label, style)

        color_legend_elements = [color_line(color, label) for color, label in zip(colors, labels)]
        color_legend_elements += [color_line(color='black', label=full_label)]
        style_legend_elements = [style_line(styles[i], name) for i, name in enumerate(self.data.scores.keys())]

        print('\n\n\n')
        print(bins)
        print(len(self.data.df.pt)/1e5)
        print(len(self.data.truth)/1e5)
        print('\n\n\n')
        masks = [self.get_mask(key, lower, upper) for (lower, upper) in bins]
        if apply_to=='fake':
            masks = [(mask | self.data.df.real) for mask in masks]  # set all real to true (== apply mask only to fakes)
        elif apply_to=="real":
            masks = [(mask | self.data.df.fake) for mask in masks]  # set all fake to true (== apply mask only to real)

        for i, (modelname, score) in enumerate(self.data.scores.items()):
            current_style = styles[i]
            for j, mask in enumerate(masks):
                current_color = colors[j]
                current_label = labels[j]
                # plot subset
                print(j, mask.sum()/1e5)
                if mask.sum()==0: continue
                
                plot_mode(axis, score, selection=mask, threshold=threshold, 
                      fifty_percent_line=fifty_percent_line, label=current_label, color=current_color, ls=current_style, **kwargs)
            # plot full dataset
            plot_mode(axis, score, selection=None, threshold=threshold, 
                      fifty_percent_line=fifty_percent_line, label=full_label, color='black', ls=current_style, **kwargs)
        

        # set legend(s)
        if mode=='normal' or mode=='classic' or mode=='thresholds':
            first_legend = axis.legend(handles=color_legend_elements, loc='upper right')
            axis.add_artist(first_legend)
            axis.legend(handles=style_legend_elements, loc='center right')
        elif mode=='ratio':
            axis.legend(handles=color_legend_elements, loc='upper right')
        elif mode=='output':
            first_legend = axis.legend(handles=color_legend_elements, loc='upper center')
            axis.add_artist(first_legend)
            axis.legend(handles=style_legend_elements, loc='upper left')
        

    def get_mask(self, key_or_array: Union[str, NDArray], minimum: Union[float, None], maximum: Union[float]) -> Mask:
        if isinstance(key_or_array, str):
            selector = self.data.df[key_or_array]
        else:
            selector = key_or_array
        sel: Mask = np.ones(selector.shape, dtype=bool)
        if minimum is not None:
            sel &= (minimum <= selector)
        if maximum is not None:
            sel &= (selector < maximum)
        else:
            return ValueError('minimum and maximum cannot both be None')
            
        return sel

