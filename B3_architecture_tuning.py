#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:43:00 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import sys
import datetime
import lib.misc as misc
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import time
import lib.preprocessing as preproc
from tensorflow.keras.layers import Input, Activation, Dense, Flatten
from tensorflow.keras.models import Model
from keras_tuner import HyperParameters, RandomSearch, BayesianOptimization
from lib.cython_impl.tools import scale_data as cscale_data
from tcn import TCN
from tcn.tcn import process_dilations
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import io
from contextlib import redirect_stdout
from tensorflow.keras.optimizers.schedules import ExponentialDecay






def start_GPU_session():
    import tensorflow as tf
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1 , 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)



def reset_TF_session():
    import tensorflow as tf
    tf.compat.v1.keras.backend.clear_session()




def generator(PARAMS, folder, file_list, batchSize):
    batch_count = 0
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])

    file_list_sp_temp = file_list['speech'].copy()
    file_list_mu_temp = file_list['music'].copy()

    batchData_sp = np.empty([], dtype=float)
    batchData_mu = np.empty([], dtype=float)

    balance_sp = 0
    balance_mu = 0

    if not os.path.exists(PARAMS['feature_opDir']+'/speech/'):
        os.makedirs(PARAMS['feature_opDir']+'/speech/')
    if not os.path.exists(PARAMS['feature_opDir']+'/music/'):
        os.makedirs(PARAMS['feature_opDir']+'/music/')

    if len(PARAMS['classes'])==3:
        np.random.shuffle(file_list['speech+music'])
        file_list_spmu_temp = file_list['speech+music'].copy()
        batchData_spmu = np.empty([], dtype=float)
        balance_spmu = 0
        if not os.path.exists(PARAMS['feature_opDir']+'/speech_music/'):
            os.makedirs(PARAMS['feature_opDir']+'/speech_music/')
        
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]
        
    while 1:
        batchData = np.empty([], dtype=float)
        
        while balance_sp<batchSize:
            if not file_list_sp_temp:
                file_list_sp_temp = file_list['speech'].copy()
            sp_fName = file_list_sp_temp.pop()
            sp_fName_path = folder + '/speech/' + sp_fName
            if not os.path.exists(sp_fName_path):
                # print(sp_fName_path, os.path.exists(sp_fName_path))
                continue         
            fv_sp = preproc.get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', None, n_fft, n_mels, featName)
            if PARAMS['frame_level_scaling']:
                fv_sp = cscale_data(fv_sp, PARAMS['mean_fold'+str(PARAMS['fold'])], PARAMS['stdev_fold'+str(PARAMS['fold'])])
            fv_sp_patches = preproc.get_feature_patches(PARAMS, fv_sp, PARAMS['W'], PARAMS['W_shift'], featName)
            if balance_sp==0:
                batchData_sp = fv_sp_patches
            else:
                batchData_sp = np.append(batchData_sp, fv_sp_patches, axis=0)
            balance_sp += np.shape(fv_sp_patches)[0]
            # print('Speech: ', batchSize, balance_sp, np.shape(batchData_sp))
            

        while balance_mu<batchSize:
            if not file_list_mu_temp:
                file_list_mu_temp = file_list['music'].copy()
            mu_fName = file_list_mu_temp.pop()
            mu_fName_path = folder + '/music/' + mu_fName
            if not os.path.exists(mu_fName_path):
                # print(mu_fName_path, os.path.exists(mu_fName_path))
                continue
            fv_mu = preproc.get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, None, n_fft, n_mels, featName)
            if PARAMS['frame_level_scaling']:
                fv_mu = cscale_data(fv_mu, PARAMS['mean_fold'+str(PARAMS['fold'])], PARAMS['stdev_fold'+str(PARAMS['fold'])])
            fv_mu_patches = preproc.get_feature_patches(PARAMS, fv_mu, PARAMS['W'], PARAMS['W_shift'], featName)
            if balance_mu==0:
                batchData_mu = fv_mu_patches
            else:
                batchData_mu = np.append(batchData_mu, fv_mu_patches, axis=0)
            balance_mu += np.shape(fv_mu_patches)[0]
            # print('Music: ', batchSize, balance_mu, np.shape(batchData_mu))

        batchData = batchData_mu[:batchSize, :] # music label=0
        batchData = np.append(batchData, batchData_sp[:batchSize, :], axis=0) # speech label=1

        balance_mu -= batchSize
        balance_sp -= batchSize

        batchData_mu = batchData_mu[batchSize:, :]            
        batchData_sp = batchData_sp[batchSize:, :]

        batchLabel = [0]*batchSize # music
        batchLabel.extend([1]*batchSize) # speech

        if len(PARAMS['classes'])==3:
            while balance_spmu<batchSize:
                if not file_list_spmu_temp:
                    file_list_spmu_temp = file_list['speech+music'].copy()
                np.random.shuffle(file_list_spmu_temp)
                spmu_info = file_list_spmu_temp.pop()
                sp_fName = spmu_info['speech']
                sp_fName_path = folder + '/speech/' + sp_fName
                mu_fName = spmu_info['music']
                mu_fName_path = folder + '/music/' + mu_fName
                target_dB = spmu_info['SMR']
                if (not os.path.exists(mu_fName_path)) or (not os.path.exists(sp_fName_path)):
                    continue
                fv_spmu = preproc.get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], sp_fName_path, mu_fName_path, target_dB, n_fft, n_mels, featName)
                if PARAMS['frame_level_scaling']:
                    fv_spmu = cscale_data(fv_spmu, PARAMS['mean_fold'+str(PARAMS['fold'])],  PARAMS['stdev_fold'+str(PARAMS['fold'])])
                fv_spmu_patches = preproc.get_feature_patches(PARAMS, fv_spmu, PARAMS['W'], PARAMS['W_shift'], featName)
                if balance_spmu==0:
                    batchData_spmu = fv_spmu_patches
                else:
                    batchData_spmu = np.append(batchData_spmu, fv_spmu_patches, axis=0)
                balance_spmu += np.shape(fv_spmu_patches)[0]
                # print('SpeechMusic: ', batchSize, balance_spmu, np.shape(batchData_spmu))
                
            # speech_music label=2
            batchData = np.append(batchData, batchData_spmu[:batchSize, :], axis=0)  
            balance_spmu -= batchSize
            batchData_spmu = batchData_spmu[batchSize:, :]            
            batchLabel.extend([2]*batchSize) # speech+music
        
        if PARAMS['Model']=='Lemaire_et_al':
            batchData = np.transpose(batchData, axes=(0,2,1)) # TCN input shape=(batch_size, timesteps, ndim)
        
        ''' Adding Normal (Gaussian) noise for data augmentation '''
        if PARAMS['data_augmentation_with_noise']:
            scale = np.random.choice([5e-3, 1e-3, 5e-4, 1e-4])
            noise = np.random.normal(loc=0.0, scale=scale, size=np.shape(batchData))
            batchData = np.add(batchData, noise)
                
        OHE_batchLabel = to_categorical(batchLabel, num_classes=len(PARAMS['classes']))
    
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(OHE_batchLabel))
        yield batchData, OHE_batchLabel
            



def get_Lemaire_model(hp):
    '''
    TCN based model architecture proposed by Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection    

    Parameters
    ----------
    hp : object
        Hyperparameters.

    Returns
    -------
    model : tensorflow.keras.models.Model
        CNN model.

    '''
    dilations = [2**nd for nd in range(hp.get('Nd'))]
    list_n_filters = [hp.get('n_filters')]*hp.get('n_layers')
    dropout_rate = np.random.uniform(0.05,0.5)
    bidirectional = True
    N_MELS = 80
    n_classes = 2
    patch_size = 68
    
    if bidirectional:
        padding = 'same'
    else:
        padding = 'causal'

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(patch_size, N_MELS))
        
    for i in range(hp.get('n_layers')):
        if i == 0:
            x = TCN(list_n_filters[i], hp.get('kernel_size'), hp.get('nb_stacks'), dilations, 'norm_relu', padding, hp.get('skip_some_connections'), dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], hp.get('kernel_size'), hp.get('nb_stacks'), dilations, 'norm_relu', padding, hp.get('skip_some_connections'), dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Flatten()(x)

    x = Dense(n_classes)(x)
    x = Activation('softmax')(x)
    output_layer = x
    
    model = Model(input_layer, output_layer)

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*hp.get('TR_STEPS'), decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    if n_classes==2:
        model.compile(loss='binary_crossentropy', metrics='accuracy', optimizer=optimizer)
    elif n_classes==3:
        model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer=optimizer)

    # print(model.summary())
    # print('Architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')
    return model




def get_tuner(opDir, method, max_trials):
    hp = HyperParameters()
    hp.Int('kernel_size', min_value=3, max_value=19, step=2) # 9
    hp.Int('Nd', min_value=3, max_value=8, step=1) # 6
    hp.Int('nb_stacks', min_value=3, max_value=10, step=1) # 8
    hp.Int('n_layers', min_value=1, max_value=4, step=1) # 4
    hp.Choice('n_filters', [8, 16, 32]) # 3
    hp.Boolean('skip_some_connections') # 2
    hp.Fixed('TR_STEPS', PARAMS['TR_STEPS'])

    if method=='RandomSearch':
        tuner = RandomSearch(
            get_Lemaire_model,
            hyperparameters = hp,
            objective = 'val_loss',
            max_trials = max_trials,
            executions_per_trial = 2,
            overwrite = False,
            directory = opDir,
            project_name = 'B3_architecture_tuning_non_causal',
            tune_new_entries = True,
            allow_new_entries = True,
            )

    elif method=='BayesianOptimization':
        tuner = BayesianOptimization(
            get_Lemaire_model,
            hyperparameters = hp,
            objective = 'val_loss',
            max_trials = max_trials,
            executions_per_trial = 2,
            overwrite = False,
            directory = opDir,
            project_name = 'B3_architecture_tuning_non_causal',
            tune_new_entries = True,
            allow_new_entries = True,
            )
    
    return tuner



def get_train_test_files(cv_file_list, foldNum):
    train_files = {}
    test_files = {}
    classes = {'music':0, 'speech':1, 'speech+music':2}
    for class_name in classes.keys():
        files = cv_file_list[class_name]['fold'+str(foldNum)]
        nTrain = int(len(files)*0.7)
        train_files[class_name] = files[:nTrain]
        test_files[class_name] = files[nTrain:]
    
    return train_files, test_files



def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            # 'folder': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Scheirer-slaney/',
            # 'folder': '/scratch/mbhattacharjee/data/musan/',
            'folder': '/home/phd/mrinmoy.bhattacharjee/data/musan/', # EEE GPU
            # 'feature_folder': './features/', 
            'feature_folder': '/home1/PhD/mrinmoy.bhattacharjee/features/',  # EEE GPU
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'TR_STEPS': 0,
            'V_STEPS': 0,
            'test_steps': 0,
            'epochs': 50,
            'batch_size': 16,
            'Model': 'Lemaire_et_al',
            'GPU_session':None,
            'classes': {0:'music', 1:'speech'}, # {0:'music', 1:'speech', 2:'speech_music'}
            'data_augmentation_with_noise': False,
            'n_fft': {'Lemaire_et_al':400},
            'n_mels': {'Lemaire_et_al':80},
            'input_shape': {'Lemaire_et_al':(68,80)},
            'W':68,
            'W_shift':68,
            'Tw': 25,
            'Ts': 10,
            'featName': {'Lemaire_et_al':'LogMelSpec'} ,
            'frame_level_scaling': False,
            'max_trials': 20,
            }
    
    PARAMS['dataset_name_train'] = list(filter(None,PARAMS['folder'].split('/')))[-1]
    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name_train'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list')

    n_classes = len(PARAMS['classes'])
    DT_SZ = 0
    for clNum in PARAMS['classes'].keys():
        classname = PARAMS['classes'][clNum]
        if classname=='speech_music':
            classname = 'speech+music'
        DT_SZ += PARAMS['cv_file_list']['total_duration'][classname] # in Hours
    DT_SZ *= 3600*1000 # in msec
    DT_SZ /= 3 # Tuning done for only one fold
    tr_frac = ((PARAMS['CV_folds']-1)/PARAMS['CV_folds'])*0.7
    vl_frac = ((PARAMS['CV_folds']-1)/PARAMS['CV_folds'])*0.3
    ts_frac = (1/PARAMS['CV_folds'])
    shft = PARAMS['W_shift']*PARAMS['Ts'] # Interval shift in milisecs
    PARAMS['TR_STEPS'] = int(np.floor(DT_SZ/shft)*tr_frac/(n_classes*PARAMS['batch_size']))
    PARAMS['V_STEPS'] = int(np.floor(DT_SZ/shft)*vl_frac/(n_classes*PARAMS['batch_size']))
    PARAMS['TS_STEPS'] = int(np.floor(DT_SZ/shft)*ts_frac/(n_classes*PARAMS['batch_size']))
    print('TR_STEPS: %d, \tV_STEPS: %d,  \tTS_STEPS: %d\n'%(PARAMS['TR_STEPS'], PARAMS['V_STEPS'], PARAMS['TS_STEPS']))
    
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    start_time = time.process_time()
        
    PARAMS['feature_opDir'] = PARAMS['feature_folder'] + '/' + PARAMS['dataset_name_train'] + '/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])
        
    PARAMS['opDir'] = './results/' + PARAMS['dataset_name_train'] + '/Baseline_HpOpt/' + PARAMS['today'] + '/' + PARAMS['Model'] + '_' + PARAMS['featName'][PARAMS['Model']] + '_' + str(len(PARAMS['classes'])) + 'classes/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
                
    misc.print_configuration(PARAMS)
                
    PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS['cv_file_list'], PARAMS['fold'])
    train_files = {}
    val_files = {}
    for classname in  PARAMS['train_files'].keys():
        files = PARAMS['train_files'][classname]
        np.random.shuffle(files)
        nTrain = int(len(files)*0.7)
        train_files[classname] = files[:nTrain]
        val_files[classname] = files[nTrain:]
        print(classname, nTrain, len(files)-nTrain)
    
    tuner = get_tuner(PARAMS['opDir'], 'BayesianOptimization', max_trials=PARAMS['max_trials']) # BayesianOptimization, RandomSearch
    
    if PARAMS['use_GPU']:
        PARAMS['GPU_session'] = start_GPU_session()

    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    csv_logger = CSVLogger(PARAMS['opDir']+'/tuner_search_log.csv')
    tuner.search(
        generator(PARAMS, PARAMS['folder'], train_files, PARAMS['batch_size']),
        steps_per_epoch = PARAMS['TR_STEPS'],
        epochs = PARAMS['epochs'], 
        validation_data = generator(PARAMS, PARAMS['folder'], val_files, PARAMS['batch_size']),
        validation_steps = PARAMS['V_STEPS'],
        # callbacks=[es, csv_logger],
        callbacks=[es],
        verbose=1,
        )
        
    weightFile = PARAMS['opDir'] + '/best_model.h5'
    architechtureFile = PARAMS['opDir'] + '/best_model.json'
    paramFile = PARAMS['opDir'] + '/best_model.npz'

    best_model = tuner.get_best_models(1)[0]
    trainingTimeTaken = time.process_time() - start_time
    learning_rate = 0.002
    
    best_model.save_weights(weightFile) # Save the weights
    with open(architechtureFile, 'w') as f: # Save the model architecture
        f.write(best_model.to_json())
    np.savez(paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], lr=learning_rate, trainingTimeTaken=trainingTimeTaken)
        
    f = io.StringIO()
    with redirect_stdout(f):
        tuner.results_summary()
    summary = f.getvalue()
    with open(PARAMS['opDir']+'/tuner_results.txt', 'w+') as f:
        f.write(summary)

    best_hparams = tuner.get_best_hyperparameters(1)[0].values
    print(best_hparams)
    with open(PARAMS['opDir']+'/best_hyperparameters.csv', 'w+') as f:
        for key in best_hparams.keys():
            f.write(key + ',' + str(best_hparams[key]) + '\n')

    if PARAMS['use_GPU']:
        reset_TF_session()
