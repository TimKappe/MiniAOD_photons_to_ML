print('starting python')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse

print('general imports done')
import myplotparams

# my functions
from mynetworks import Patches, PatchEncoder, mlp
from mynetworks import plot_patches, resize_images

from mymodules import plot_training
from mymodules import plot_output

from myparameters import Parameters
from myparameters import data_from_params, weights_from_params
from myparameters import build_vit_from_params
from myparameters import rescale_multiple, split_data, split_multiple
print('my imports done')

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask, Sparse
from mytypes import Callback, Layer
print('typing imports done')

from rechit_handler import RechitHandler


# shorthands
models = keras.models
layers = keras.layers

print('starting proper code:')
parser = argparse.ArgumentParser(description='train vit with hyperparameters from parameterfile')
parser.add_argument('parameterfile', help='file to be read')
paramfile = parser.parse_args().parameterfile
print(paramfile)
params = Parameters(load=paramfile)
print('Parameters:')
print(params)

### load and prepare labels, weights and other inputs (rechits get loaded in the RecHitsHandler)
df: pd.DataFrame = pd.read_pickle(params['dataframefile'])
y_train, y_test = split_data(params, df.real.to_numpy(dtype=int))

needs_scaling = []
no_scaling = []
for key in params['other_inputs']:
    var: NDArray = df[key].to_numpy()
    if key=='pt':
        var = np.log(var)
    if key in ['converted', 'convertedOneLeg']: 
        no_scaling += [var.astype(int)] 
    else: 
        needs_scaling += [var]

weights_train, weights_test = weights_from_params(params, selection=None)  # this is the training set only

scaled_inputs_train, scaled_inputs_test = rescale_multiple(params, needs_scaling, weights_train)
non_scaled_train, non_scaled_test = split_multiple(params, no_scaling)

other_train_inputs = np.column_stack(scaled_inputs_train + non_scaled_train)
other_test_inputs = np.column_stack(scaled_inputs_test + non_scaled_test)


########################################################################
# create rechitshandler
rechitfile: Filename = params['rechitfile']
try:
    params['batch_size'] = params['fit_params']['batch_size']
except:
    ReferenceError
TrainHandler = RechitHandler(rechitfile, other_train_inputs, y_train, weights_train, 
                             params['batch_size'], params['image_size'], which_set='train')
ValHandler = RechitHandler(rechitfile, other_train_inputs, y_train, weights_train, 
                           params['batch_size'], params['image_size'], which_set='val')
TestHandler = RechitHandler(rechitfile, other_test_inputs, y_test, weights_test, 
                           params['batch_size'], params['image_size'], which_set='test')



########################################################################

########################################################################
### Callbacks
callbacks: List[Callback] = []
if params['use_checkpointing']:
    checkpointfile = params['modeldir'] + params['modelname'] + '_checkpoints'
    callbacks += [keras.callbacks.ModelCheckpoint(checkpointfile, **params['checkpointing'])]
if params['use_earlystopping']:
    callbacks += [keras.callbacks.EarlyStopping(**params['earlystopping'])]
if params['use_reduce_lr']:
    callbacks += [keras.callbacks.ReduceLROnPlateau(**params['reduce_lr'])]

# todo use on_epoch_end for custom (correct) validation loss
#from tensorflow.keras import backend as K
#class printlearningrate(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        optimizer = self.model.optimizer
#        lr = K.eval(optimizer.lr)
#        Epoch_count = epoch + 1
#        print('\n', "Epoch:", Epoch_count, ', LR: {:.2f}'.format(lr))


########################################################################
model = build_vit_from_params(params)
optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              weighted_metrics=['accuracy'])

model.summary()

# todo make fit_params a dict in Parameters
# history = model.fit(TrainHandler, 
history = model.fit(TrainHandler, 
                    validation_data=ValHandler,
                    callbacks=callbacks,
                    epochs=params['fit_params']['epochs'],
                    verbose=2
                    )

### save model and history
modelsavefile = params['modeldir'] + params['modelname'] + '.keras'
historyfile = params['modeldir'] + params['modelname'] + '_history.npy'
model.save(modelsavefile)
np.save(historyfile, history.history)
print('model saved as', modelsavefile)
print('history saved as', historyfile)



##################################################################
test_loss, test_acc = model.evaluate(TestHandler, verbose=params['fit_params']['verbose'])
print('test_accuracy =', test_acc)

### plot training curves
figname: str = params['modeldir'] + params['modelname'] + '_training.png'
plot_training(history.history, test_acc, savename=figname)  # info printed inside function

##############################################################################
### calculate output
# TODO do this properly later
def sparse_to_dense(sparse: Sparse) -> NDArray:
    dense = np.zeros((int(0.2*len(df)), params['image_size'], params['image_size']), dtype=np.float32)
    values, indices = sparse[0], sparse[1:]
    dense[indices] = values
    return dense

# x_test_dense = sparse_to_dense(TestHandler.get_sparse_rechits())


y_pred: NDArray = model.predict(TestHandler, verbose=params['fit_params']['verbose']).flatten()  # output is shape (..., 1)
savename: Filename = params['modeldir'] + params['modelname'] + '_pred.npy'
np.save(savename, y_pred)
print(f'INFO: prediction saves as {savename}')


### plot output
real: Mask = y_test.astype(bool)
binning: Tuple[int, float, float] = (int(1/params['output_binwidth']), 0., 1.)

fig, ax = plt.subplots(figsize=(10, 8))
plot_output(ax, y_pred, real, binning)  # real and fake outputs are normalized separately
ax.set_title(f'Output {params["ModelName"]}')
plt.tight_layout()

outputfile: Filename = params['modeldir'] + params['modelname'] + '_output.png'
fig.savefig(outputfile)
print(f'INFO: output saved as {outputfile}')
##############################################################################
### save parameters used
paramfile: Filename = params['modelname'] + '_params.txt'
params.save(paramfile)  # infoprint in save function



print('FINISHED')
