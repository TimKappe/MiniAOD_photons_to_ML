import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import myplotparams

from myparameters import Parameters

from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray
from mytypes import Filename, Mask, Layer, Callback





base = Parameters(
    ## data
    other_inputs = ['pt', 'eta', 'rho', 'HoE', 'I_tr', 'hcalIso', 'converted', 'convertedOneLeg'],
    dataframefile='data/data_barrel_pre.pkl',
    rechitfile='data/rechits_barrel_pre.npy',
    weightfile='data/weights_barrel_real.npy',
    preselectionfile=None,
    barrel_only=True,
    test_split=0.2,
    use_log=True,
    input_shape=[32, 32, 3],
    image_size=32,

    ## for fitting
    fit_params=dict(
        validation_split=0.25, epochs=500, batch_size=4096, verbose=2),  # val_split0.25 means split of 60/20/20,
                                                                         # need to account for testsplit first
    validation_split=0.25, 
    learning_rate=1e-3,
    epochs=500,
    batch_size=4096,
    output_binwidth=0.05,
)

base_vit = base + Parameters(
    # modelname stuff
    modeldir='models/',
    modelname='vit',
    ModelName='ViT',

    # build model stuff
    patch_size=4,
    # num_patches wird automatisch gesetzt
    projection_dim=16,
    num_heads=4,
    transformer_units=[16, 16],  # last one needs to be projection_dim
    transformer_layers=4,
    mlp_head_units=[32, 32, 16],
    dropout_rate=0.0,
    final_dropout=0.0,
    layer_norm_epsilon=1e-6,

    ## callbacks
    use_earlystopping=True,
    use_reduce_lr=False,
    use_checkpointing=False,
    earlystopping=dict(
        monitor='val_loss', min_delta=0, patience=30, mode='auto', verbose=2, restore_best_weights=True),
    reduce_lr=dict(
        monitor='val_loss', factor=0.9, patience=20, verbose=2),
    checkpointing=dict(monitor='val_accuracy', save_best_only=True),  # checkpointfile is defined in script
                                                                            # from other params
)

base_cnn = base + Parameters(
    # modelname stuff
    modeldir='models/',
    modelname='cnn',
    ModelName='CNN',

    ## for fitting
    fit_params=dict(
        validation_split=0.25, epochs=500, batch_size=4096, verbose=2),  # val_split0.25 means split of 60/20/20,
                                                                         # need to account for testsplit first
    learning_rate=1e-3,
    output_binwidth=0.05,

    ## build model stuff
    input_shape=[11, 11],
    image_size=11,
    patch_size=4,
    # num_patches wird automatisch gesetzt
    projection_dim=16,
    num_heads=4,
    transformer_units=[16, 16],  # last one needs to be projection_dim
    transformer_layers=4,
    mlp_head_units=[32, 32, 16],
    dropout_rate=0.0,
    final_dropout=0.0,
    layer_norm_epsilon=1e-6,

    ## callbacks
    use_earlystopping=True,
    use_reduce_lr=False,
    use_checkpointing=False,
    earlystopping=dict(
        monitor='val_loss', min_delta=0, patience=30, mode='auto', verbose=2, restore_best_weights=True),
    reduce_lr=dict(
        monitor='val_loss', factor=0.9, patience=25, verbose=2),
    checkpointing=dict(monitor='val_accuracy', save_best_only=True),  # checkpointfile is defined in script
                                                                            # from other params
)

base_vit.save('models/vit_base.json')
base_cnn.save('models/cnn_base.json')


patchsize = Parameters(load='models/vit_base.json')
patchsize['modelname'] = 'vit_patch1'
patchsize['patch_size'] = 1
patchsize['dataframefile'] = 'data/data_high.pkl'
patchsize['rechitfile'] = 'data/rechits_11x11_high.npz'
patchsize['weightfile'] = 'data/weights_real.npy'
patchsize.save()


vit32 = Parameters(load='models/vit_base.json')
vit32['modelname'] = 'vit32'
vit32['dataframefile'] = 'data/data_32x32_high.pkl'
vit32['rechitfile']='data/rechits_32x32_high.npz'
vit32['weightfile']='data/weights_32x32_real.npy'
vit32['input_shape']=[32, 32, 3]
vit32['image_size']=32
vit32.save()

for i in range(6):
    vit32 = Parameters(load='models/vit32.json')
    vit32['modelname'] = f'vit32_patch{i+1}'
    vit32['patch_size'] = i+1
    vit32['earlystopping']['patience'] = (i+1)*10
    vit32.save()

vit11 = Parameters(load='models/vit_base.json')
vit11['modelname'] = 'vit_11'
vit11['dataframefile'] = 'data/data_11x11_high.pkl'
vit11['rechitfile']='data/rechits_11x11_high.npz'
vit11['weightfile']='data/weights_11x11_real.npy'
vit11['input_shape']=[11, 11, 3]
vit11['image_size']=11
vit11.save()


large_dense = Parameters(load='models/vit_base.json')
large_dense['modelname'] = 'vit_dense_wide'
large_dense['dataframefile'] = 'data/data_32x32_high.pkl'
large_dense['rechitfile']='data/rechits_32x32_high.npz'
large_dense['weightfile']='data/weights_32x32_real.npy'
large_dense['mlp_head_units']=[512]
large_dense.save()

large_dense = Parameters(load='models/vit_base.json')
large_dense['modelname'] = 'vit_dense_large'
large_dense['dataframefile'] = 'data/data_32x32_high.pkl'
large_dense['rechitfile']='data/rechits_32x32_high.npz'
large_dense['weightfile']='data/weights_32x32_real.npy'
large_dense['mlp_head_units']=[512, 64, 8]
large_dense.save()

large_dense = Parameters(load='models/vit_base.json')
large_dense['modelname'] = 'vit_dense_mid'
large_dense['dataframefile'] = 'data/data_32x32_high.pkl'
large_dense['rechitfile']='data/rechits_32x32_high.npz'
large_dense['weightfile']='data/weights_32x32_real.npy'
large_dense['mlp_head_units']=[128, 64, 16]
large_dense['learning_rate']=1e-4
large_dense.save()




# base = Parameters(
#     ## data and modelnamesimage_size
#     modeldir='models/',
#     modelname='vit',
#     ModelName='ViT',
#     dataframefile='data/new_data_pre_barrel.pkl',
#     rechitfile='data/new_rechits_pre_barrel.npy',
#     weightfile='data/weights_real_barrel.npy',
#     preselectionfile=None,
#     barrel_only=True,
#     test_split=0.2,
#     use_log=True,

#     ## for fitting
#     fit_params=dict(
#         validation_split=0.25, epochs=500, batch_size=4096, verbose=2),  # val_split0.25 means split of 60/20/20,
#                                                                          # need to account for testsplit first
#     learning_rate=1e-3,
#     output_binwidth=0.05,

#     ## build model stuff
#     input_shape=[11, 11, 3],
#     image_size=11,
#     patch_size=4,
#     # num_patches wird automatisch gesetzt
#     projection_dim=16,
#     num_heads=4,
#     transformer_units=[16, 16],  # last one needs to be projection_dim
#     transformer_layers=4,
#     mlp_head_units=[32, 32, 16],
#     dropout_rate=0.0,
#     final_dropout=0.0,
#     layer_norm_epsilon=1e-6,

#     ## callbacks
#     use_earlystopping=True,
#     use_reduce_lr=False,
#     use_checkpointing=False,
#     earlystopping=dict(
#         monitor='val_loss', min_delta=0, patience=10, mode='auto', verbose=2, restore_best_weights=True),
#     reduce_lr=dict(
#         monitor='val_loss', factor=0.9, patience=20, verbose=2),
#     checkpointing=dict(monitor='val_accuracy', save_best_only=True),  # checkpointfile is defined in script
#                                                                             # from other params
# )

# params = Parameters(load='base.json')
# params['final_dropout'] = 0.5
# params['modelname'] = 'vit_dropout_final'
# params.save()

# params = Parameters(load='base.json')
# params['dropout_rate'] = 0.3
# params['modelname'] = 'vit_dropout'
# params.save()

# params = Parameters(load='base.json')
# params['reweight'] = 'weight_real'
# params['modelname'] = 'vit_weight_real'
# params.save()

# params = Parameters(load='base.json')
# params['learning_rate'] = 1e-5
# params['modelname'] = 'vit_lr_small'
# params.save()






# layers_ = [2, 4, 8]
# hea#ds = [2, 4, 8, 16, 32]
# # fast and ugly
# def combinations(a, b):
#     out = []
#     for aa in a:
#         out += [(aa, bb) for bb in b]
#     return out

# # def combinations(list1, list2) -> NDArray:
# #     """returns NDArray with tuples of the combinations of input list"""
# #     arr = np.zeros((len(list1) * len(list2), 2), dtype=int)
# #     i = 0
# #     for _, val1 in enumerate(list1):
# #         for _, val2 in enumerate(list2):
# #             arr[i] = (val1, val2)
# #             i += 1
# #     return arr
# # combinations(layers_, heads)

# name = base['modelname']
# for layercount, headcount in combinations(layers_, heads):
#     modelnames = name + f'_{layercount}_{headcount}'
#     base['modelname'] = modelnames
#     base['ModelName'] = modelnames
#     base['num_heads'] = layercount
#     base['transformer_layers'] = headcount
#     outfile: Filename = base['modeldir'] + base['modelname'] + '.json'
#     base.save(outfile)
#     print(f"INFO: parameters saved as {outfile}")


print('FINISHED')





