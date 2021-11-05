#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:51:23 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import sys
import datetime
import lib.misc as misc
from lib.baseline_architectures import get_Lemaire_model
from lib.proposed_architectures import get_Lemaire_MTL_model, get_Lemaire_Cascaded_MTL_model
from lib.proposed_architectures import get_Doukhan_MTL_model, get_Papakostas_MTL_model, get_Jang_MTL_model
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
import time
import lib.preprocessing as preproc
from lib.cython_impl.tools import scale_data as cscale_data
import tensorflow as tf
from lib.cython_impl.tools import get_data_statistics as cget_data_statistics
from tensorflow.keras.optimizers.schedules import ExponentialDecay




def start_GPU_session():
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
        batchData_spmu_target_dB = np.empty([], dtype=float)
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

            if PARAMS['skewness_vector']:
                # Added 06-Oct-21
                # if ('HarmPerc' in featName) or  ('Harm' in featName) or  ('Perc' in featName):
                #     nDim = np.shape(fv_sp_patches)[1]
                #     if PARAMS['skewness_vector']=='Row':
                #         fv_sp_patches = cget_data_statistics(fv_sp_patches[:,:int(nDim/2),:], axis=1)
                #         fv_sp_patches = np.expand_dims(fv_sp_patches, axis=2)
                #     elif PARAMS['skewness_vector']=='Col':
                #         fv_sp_patches = cget_data_statistics(fv_sp_patches[:,int(nDim/2):,:], axis=0)
                #         fv_sp_patches = np.expand_dims(fv_sp_patches, axis=1)
                # else:
                if PARAMS['skewness_vector']=='Row':
                    fv_sp_patches = cget_data_statistics(fv_sp_patches, axis=1)
                    fv_sp_patches = np.expand_dims(fv_sp_patches, axis=2)
                elif PARAMS['skewness_vector']=='Col':
                    fv_sp_patches = cget_data_statistics(fv_sp_patches, axis=0)
                    fv_sp_patches = np.expand_dims(fv_sp_patches, axis=1)

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

            if PARAMS['skewness_vector']:
                # Added 06-Oct-21
                # if ('HarmPerc' in featName) or  ('Harm' in featName) or  ('Perc' in featName):
                #     nDim = np.shape(fv_mu_patches)[1]
                #     if PARAMS['skewness_vector']=='Row':
                #         fv_mu_patches = cget_data_statistics(fv_mu_patches[:,:int(nDim/2),:], axis=1)
                #         fv_mu_patches = np.expand_dims(fv_mu_patches, axis=2)
                #     elif PARAMS['skewness_vector']=='Col':
                #         fv_mu_patches = cget_data_statistics(fv_mu_patches[:,int(nDim/2):,:], axis=0)
                #         fv_mu_patches = np.expand_dims(fv_mu_patches, axis=1)
                # else:                
                if PARAMS['skewness_vector']=='Row':
                    fv_mu_patches = cget_data_statistics(fv_mu_patches, axis=1)
                    fv_mu_patches = np.expand_dims(fv_mu_patches, axis=2)
                elif PARAMS['skewness_vector']=='Col':
                    fv_mu_patches = cget_data_statistics(fv_mu_patches, axis=0)
                    fv_mu_patches = np.expand_dims(fv_mu_patches, axis=1)

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
        batchLabel_smr = np.ones((3*batchSize,2))
        batchLabel_smr[:batchSize, :] = np.repeat(np.array([1, 0], ndmin=2), batchSize, axis=0) # music
        batchLabel_smr[batchSize:2*batchSize] = np.repeat(np.array([0, 1], ndmin=2), batchSize, axis=0) # speech

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

                if PARAMS['skewness_vector']:
                    # Added 06-Oct-21
                    # if ('HarmPerc' in featName) or  ('Harm' in featName) or  ('Perc' in featName):
                    #     nDim = np.shape(fv_spmu_patches)[1]
                    #     if PARAMS['skewness_vector']=='Row':
                    #         fv_spmu_patches = cget_data_statistics(fv_spmu_patches[:,:int(nDim/2),:], axis=1)
                    #         fv_spmu_patches = np.expand_dims(fv_spmu_patches, axis=2)
                    #     elif PARAMS['skewness_vector']=='Col':
                    #         fv_spmu_patches = cget_data_statistics(fv_spmu_patches[:,int(nDim/2):,:], axis=0)
                    #         fv_spmu_patches = np.expand_dims(fv_spmu_patches, axis=1)
                    # else:
                    if PARAMS['skewness_vector']=='Row':
                        fv_spmu_patches = cget_data_statistics(fv_spmu_patches, axis=1)
                        fv_spmu_patches = np.expand_dims(fv_spmu_patches, axis=2)
                    elif PARAMS['skewness_vector']=='Col':
                        fv_spmu_patches = cget_data_statistics(fv_spmu_patches, axis=0)
                        fv_spmu_patches = np.expand_dims(fv_spmu_patches, axis=1)

                if balance_spmu==0:
                    batchData_spmu = fv_spmu_patches
                    batchData_spmu_target_dB = np.array([target_dB]*np.shape(fv_spmu_patches)[0])
                else:
                    batchData_spmu = np.append(batchData_spmu, fv_spmu_patches, axis=0)
                    batchData_spmu_target_dB = np.append(batchData_spmu_target_dB, np.array([target_dB]*np.shape(fv_spmu_patches)[0]))
                balance_spmu += np.shape(fv_spmu_patches)[0]
                # print('SpeechMusic: ', batchSize, balance_spmu, np.shape(batchData_spmu))
                
            # speech_music label=2
            batchData = np.append(batchData, batchData_spmu[:batchSize, :], axis=0)  
            balance_spmu -= batchSize
            batchData_spmu = batchData_spmu[batchSize:, :]            
            batchLabel.extend([2]*batchSize) # speech+music
            label_idx = 2*batchSize
            for i in range(batchSize):
                if batchData_spmu_target_dB[i]>=0:
                    batchLabel_smr[label_idx] = np.array([1/np.power(10,(batchData_spmu_target_dB[i]/10)), 1], ndmin=2)
                else:
                    batchLabel_smr[label_idx] = np.array([1, np.power(10,(batchData_spmu_target_dB[i]/10))], ndmin=2)
                label_idx += 1
            batchData_spmu_target_dB = batchData_spmu_target_dB[batchSize:]
        
        if 'Lemaire_et_al' in PARAMS['Model']:
            batchData = np.transpose(batchData, axes=(0,2,1)) # TCN input shape=(batch_size, timesteps, ndim)
        
        ''' Adding Normal (Gaussian) noise for data augmentation '''
        if PARAMS['data_augmentation_with_noise']:
            scale = np.random.choice([5e-3, 1e-3, 5e-4, 1e-4])
            noise = np.random.normal(loc=0.0, scale=scale, size=np.shape(batchData))
            batchData = np.add(batchData, noise)
                            
        OHE_batchLabel = to_categorical(batchLabel, num_classes=len(PARAMS['classes']))

        '''
        Speech Nonspeech
        '''
        batchLabel_sp_nsp = np.copy(batchLabel)
        batchLabel_sp_nsp[:batchSize] = 0
        batchLabel_sp_nsp[batchSize:2*batchSize] = 1
        batchLabel_sp_nsp[2*batchSize:] = 0

        '''
        Music Nonmusic
        '''
        batchLabel_mu_nmu = np.copy(batchLabel)
        batchLabel_mu_nmu[:batchSize] = 1
        batchLabel_mu_nmu[batchSize:2*batchSize] = 0
        batchLabel_mu_nmu[2*batchSize:] = 0

        batchLabel_MTL = {'R': batchLabel_smr, 'S': batchLabel_sp_nsp, 'M': batchLabel_mu_nmu, '3C': OHE_batchLabel}
    
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(OHE_batchLabel))
        
        if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
            yield batchData, batchLabel_MTL
        else:
            yield batchData, OHE_batchLabel
            



def train_model(PARAMS, model, weightFile, logFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.process_time()

    SPE = PARAMS['TR_STEPS']
    SPE_val = PARAMS['V_STEPS']
    print('SPE: ', SPE, SPE_val)
    
    train_files = {}
    val_files = {}
    for classname in  PARAMS['train_files'].keys():
        files = PARAMS['train_files'][classname]
        np.random.shuffle(files)
        nTrain = int(len(files)*0.7)
        train_files[classname] = files[:nTrain]
        val_files[classname] = files[nTrain:]
        print(classname, nTrain, len(files)-nTrain)
    
    # Train the model
    History = model.fit(
            generator(PARAMS, PARAMS['folder'], train_files, PARAMS['batch_size']),
            steps_per_epoch = SPE,
            validation_data = generator(PARAMS, PARAMS['folder'], val_files, PARAMS['batch_size']), 
            validation_steps = SPE_val,
            epochs=PARAMS['epochs'], 
            verbose=1,
            callbacks=[csv_logger, es, mcp],
            # callbacks=[csv_logger, mcp],
            )

    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History






def perform_training(PARAMS):
    modelName = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    
    print('paramFile: ', paramFile)
    epochs = PARAMS['epochs']
    batch_size = PARAMS['batch_size']
    
    if not os.path.exists(paramFile):
        if PARAMS['Model']=='Lemaire_et_al':
            model, learning_rate = get_Lemaire_model(
                TR_STEPS=PARAMS['TR_STEPS'],  
                N_MELS=PARAMS['input_shape'][PARAMS['Model']][1], 
                n_classes=len(PARAMS['classes']),
                patch_size=PARAMS['input_shape'][PARAMS['Model']][0],
                )
        elif PARAMS['Model']=='Lemaire_et_al_MTL':
            model, learning_rate = get_Lemaire_MTL_model(
                TR_STEPS=PARAMS['TR_STEPS'],
                N_MELS=PARAMS['input_shape'][PARAMS['Model']][1], 
                n_classes=len(PARAMS['classes']),
                patch_size=PARAMS['input_shape'][PARAMS['Model']][0],
                loss_weights=PARAMS['loss_weights'],
                )
        elif PARAMS['Model']=='Lemaire_et_al_Cascaded_MTL':
            model, learning_rate = get_Lemaire_Cascaded_MTL_model(
                TR_STEPS=PARAMS['TR_STEPS'],
                N_MELS=PARAMS['input_shape'][PARAMS['Model']][1], 
                n_classes=len(PARAMS['classes']),
                patch_size=PARAMS['input_shape'][PARAMS['Model']][0],
                )
        elif PARAMS['Model']=='Doukhan_et_al_MTL':
            model, learning_rate = get_Doukhan_MTL_model(PARAMS)
        elif PARAMS['Model']=='Papakostas_et_al_MTL':
            model, learning_rate = get_Papakostas_MTL_model(PARAMS)
        elif PARAMS['Model']=='Jang_et_al_MTL':
            model, learning_rate = get_Jang_MTL_model(PARAMS)

        misc.print_model_summary(PARAMS['opDir'] + '/model_summary.txt', model)
            
        model, trainingTimeTaken, History = train_model(PARAMS, model, weightFile, logFile)
        
        if PARAMS['save_flag']:
            model.save_weights(weightFile) # Save the weights
            with open(architechtureFile, 'w') as f: # Save the model architecture
                f.write(model.to_json())
            np.savez(paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], lr=learning_rate, trainingTimeTaken=trainingTimeTaken)
        print('CNN model trained.')
    else:
        epochs = np.load(paramFile)['epochs']
        batch_size = np.load(paramFile)['batch_size']
        learning_rate = np.load(paramFile)['lr']
        trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        optimizer = optimizers.Adam(lr=learning_rate)
        with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
            model = model_from_json(f.read())
        model.load_weights(weightFile) # Load weights into the new model

        if PARAMS['Model']=='Lemaire_et_al':
            initial_learning_rate = 0.002
            lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1, decay_rate=0.1)    
            optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)
            model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer=optimizer)

        elif PARAMS['Model']=='Lemaire_et_al_MTL':
            initial_learning_rate = 0.002
            lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1, decay_rate=0.1)    
            optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)
            model.compile(
                loss={'R':'mean_squared_error', 'S': 'binary_crossentropy', 'M': 'binary_crossentropy', '3C': 'categorical_crossentropy'}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )

        elif PARAMS['Model']=='Lemaire_et_al_Cascaded_MTL':
            initial_learning_rate = 0.002
            lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1, decay_rate=0.1)    
            optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)
            model.compile(
                loss={'R':'mean_squared_error', 'S': 'binary_crossentropy', 'M': 'binary_crossentropy', '3C': 'categorical_crossentropy'}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )

        elif PARAMS['Model']=='Doukhan_et_al_MTL':
            learning_rate = 0.0001
            optimizer = optimizers.Adam(lr=learning_rate)    
            model.compile(
                loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )

        elif PARAMS['Model']=='Papakostas_et_al_MTL':
            initial_learning_rate = 0.001
            lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=700, decay_rate=0.1)    
            optimizer = optimizers.SGD(learning_rate=lr_schedule)    
            model.compile(
                loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )

        elif PARAMS['Model']=='Jang_et_al_MTL':
            learning_rate = 0.001
            optimizer = optimizers.Adam(lr=learning_rate)
            model.compile(
                loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )
            
        print('CNN model exists! Loaded. Training time required=',trainingTimeTaken)
      
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params





def test_file_wise_generator(PARAMS, file_name_sp, file_name_mu, target_dB):
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]

    if file_name_mu=='':
        fv = preproc.get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], file_name_sp, '', None, n_fft, n_mels, featName, save_feat=False)
    elif file_name_sp=='':
        fv = preproc.get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', file_name_mu, None, n_fft, n_mels, featName, save_feat=False)
    else:
        fv = preproc.get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], file_name_sp, file_name_mu, target_dB, n_fft, n_mels, featName, save_feat=False)
        
    if PARAMS['frame_level_scaling']:
        fv = cscale_data(fv, PARAMS['mean_fold'+str(PARAMS['fold'])], PARAMS['stdev_fold'+str(PARAMS['fold'])])

    batchData = preproc.get_feature_patches(PARAMS, fv, PARAMS['W'], 68, featName)
    if PARAMS['skewness_vector']:
        if PARAMS['skewness_vector']=='Row':
            batchData = cget_data_statistics(batchData, axis=1)
            batchData = np.expand_dims(batchData, axis=2)
        elif PARAMS['skewness_vector']=='Col':
            batchData = cget_data_statistics(batchData, axis=0)
            batchData = np.expand_dims(batchData, axis=1)

    if 'Lemaire_et_al' in PARAMS['Model']:
        batchData = np.transpose(batchData, axes=(0,2,1)) # TCN input shape=(batch_size, timesteps, ndim)

    numLab = np.shape(batchData)[0]
    
    if file_name_mu=='': # speech
        batchLabel = np.array([1]*numLab)
    elif file_name_sp=='': # music
        batchLabel = np.array([0]*numLab)
    else: # speech+music
        batchLabel = np.array([2]*numLab)
    OHE_batchLabel = to_categorical(batchLabel, num_classes=len(PARAMS['classes']))
    
    return batchData, OHE_batchLabel



def test_model(PARAMS, Train_Params, target_dB):
    PtdLabels = np.empty([])
    GroundTruth = np.empty([])
    Predictions = np.empty([])

    startTime = time.process_time()
    if target_dB==None:
        # class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        for classname in ['music', 'speech']:
            files = PARAMS['test_files'][classname]
            for fl in files:
                fName = PARAMS['folder'] + '/'+ classname + '/' + fl
                if not os.path.exists(fName):
                    continue
                if classname=='speech':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, fName, '', None)
                elif classname=='music':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, '', fName, None)
                
                if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
                    sp_pred, mu_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
                    sp_pred_lab = np.array((sp_pred>=0.5).astype(int))
                    acc_sp = np.round(np.sum(sp_pred_lab==1)*100/np.shape(smr_pred)[0],2)
                    mu_pred_lab = np.array((mu_pred>=0.5).astype(int))
                    acc_mu = np.round(np.sum(mu_pred_lab==1)*100/np.shape(smr_pred)[0],2)
                    # spmu_idx = np.squeeze(np.where(sp_pred_lab.flatten()==mu_pred_lab.flatten()))
                    # sp_idx = np.squeeze(np.where((sp_pred_lab.flatten()-mu_pred_lab.flatten())>0))
                    # mu_idx = np.squeeze(np.where((mu_pred_lab.flatten()-sp_pred_lab.flatten())>0))
                    # pred_lab = np.zeros(np.shape(smr_pred)[0])
                    # pred_lab[spmu_idx] = 2
                    # pred_lab[sp_idx] = 1
                    # pred_lab[mu_idx] = 0
                    # pred = smr_pred
                    pred_lab = np.argmax(pred, axis=1)
                    # print(len(sp_idx), len(mu_idx), len(spmu_idx), np.shape(smr_pred), np.unique(pred_lab, return_counts=True))
                else:
                    pred = Train_Params['model'].predict(x=batchData)
                    pred_lab = np.argmax(pred, axis=1)

                if np.size(Predictions)<=1:
                    Predictions = pred
                    PtdLabels = pred_lab
                    if classname=='speech':
                        GroundTruth = np.array([1]*np.shape(pred)[0])
                    elif classname=='music':
                        GroundTruth = np.array([0]*np.shape(pred)[0])
                else:
                    Predictions = np.append(Predictions, pred, 0)
                    PtdLabels = np.append(PtdLabels, pred_lab)
                    if classname=='speech':
                        GroundTruth = np.append(GroundTruth, np.array([1]*np.shape(pred)[0]))
                    elif classname=='music':
                        GroundTruth = np.append(GroundTruth, np.array([0]*np.shape(pred)[0]))
                
                print(target_dB, 'dB\t', classname, 'pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), end='\t', flush=True)
                if classname=='speech':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==1)*100/len(pred_lab), 4)
                    if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
                        acc_fl = [acc_sp, acc_mu, acc_fl]
                    acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
                    print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', acc_fl, acc_all)
                elif classname=='music':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==0)*100/len(pred_lab), 4)
                    if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
                        acc_fl = [acc_sp, acc_mu, acc_fl]
                    acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
                    print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', acc_fl, acc_all)

    if len(PARAMS['classes'])==3:
        files_spmu = PARAMS['test_files']['speech+music']
        fl_count = 0
        for spmu_info in files_spmu:
            fl_count += 1
            fl_sp = spmu_info['speech']
            fl_mu = spmu_info['music']
            fName_sp = PARAMS['folder'] + '/speech/' + fl_sp
            fName_mu = PARAMS['folder'] + '/music/' + fl_mu
            if target_dB==None:
                # Annotated SMR is not used in the testing function if target_dB 
                # is None so that the performance can be tested at specific 
                # SMR values
                batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, fName_mu, spmu_info['SMR'])
            else:
                batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, fName_mu, target_dB)

            if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
                sp_pred, mu_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
                sp_pred_lab = np.array((sp_pred>0.5).astype(int))
                acc_sp = np.round(np.sum(sp_pred_lab==1)*100/np.shape(smr_pred)[0],2)
                mu_pred_lab = np.array((mu_pred>0.5).astype(int))
                acc_mu = np.round(np.sum(mu_pred_lab==1)*100/np.shape(smr_pred)[0],2)
                # spmu_idx = np.squeeze(np.where(sp_pred_lab.flatten()==mu_pred_lab.flatten()))
                # sp_idx = np.squeeze(np.where((sp_pred_lab.flatten()-mu_pred_lab.flatten())>0))
                # mu_idx = np.squeeze(np.where((mu_pred_lab.flatten()-sp_pred_lab.flatten())>0))
                # pred_lab = np.zeros(np.shape(smr_pred)[0])
                # pred_lab[spmu_idx] = 2
                # pred_lab[sp_idx] = 1
                # pred_lab[mu_idx] = 0
                # pred = smr_pred
                pred_lab = np.argmax(pred, axis=1)
                # print(len(sp_idx), len(mu_idx), len(spmu_idx), np.shape(smr_pred), np.unique(pred_lab, return_counts=True))
            else:
                pred = Train_Params['model'].predict(x=batchData)
                pred_lab = np.argmax(pred, axis=1)

            if np.size(Predictions)<=1:
                Predictions = pred
                GroundTruth = np.array([2]*np.shape(pred)[0])
                PtdLabels = pred_lab
            else:
                Predictions = np.append(Predictions, pred, 0)
                GroundTruth = np.append(GroundTruth, np.array([2]*np.shape(pred)[0]))
                PtdLabels = np.append(PtdLabels, pred_lab)
            acc_fl = np.round(np.sum(np.array(pred_lab)==2)*100/len(pred_lab), 4)
            if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
                acc_fl = [acc_sp, acc_mu, acc_fl]
            acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
            if target_dB==None:
                print(fl_count, '/', len(files_spmu), spmu_info['SMR'], 'dB\tspeech_music pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), fl_sp, fl_mu, np.shape(batchData), ' acc=', acc_fl, acc_all)
            else:
                print(fl_count, '/', len(files_spmu), target_dB, 'dB\tspeech_music pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), fl_sp, fl_mu, np.shape(batchData), ' acc=', acc_fl, acc_all)

    testingTimeTaken = time.process_time() - startTime
    print('Time taken for model testing: ',testingTimeTaken)
    labels = [key for key in PARAMS['classes'].keys()]
    ConfMat, precision, recall, fscore = misc.getPerformance(PtdLabels, GroundTruth, labels)
    print(ConfMat)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('fscore: ', fscore)
    
    return ConfMat, precision, recall, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken





def perform_testing(PARAMS, Train_Params):
    ConfMat, precision, recall, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken = test_model(PARAMS, Train_Params, None)
    Test_Params = {}
    Test_Params['testingTimeTaken_annot'] = testingTimeTaken
    Test_Params['ConfMat_annot'] = ConfMat
    Test_Params['precision_annot'] = precision
    Test_Params['recall_annot'] = recall
    Test_Params['fscore_annot'] = fscore
    Test_Params['PtdLabels_test_annot'] = PtdLabels
    Test_Params['Predictions_test_annot'] = Predictions
    Test_Params['GroundTruth_test_annot'] = GroundTruth
    
    if PARAMS['dB_wise_test']:
        if len(PARAMS['classes'])==3:
            PtdLabels_All = []
            GroundTruths_All = []
            for target_dB in PARAMS['test_SMR_levels']:    
                ConfMat, precision, recall, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken = test_model(PARAMS, Train_Params, target_dB)
                PtdLabels_All.extend(PtdLabels)
                GroundTruths_All.extend(GroundTruth)
                Test_Params['testingTimeTaken_'+str(target_dB)+'dB'] = testingTimeTaken
                Test_Params['ConfMat_'+str(target_dB)+'dB'] = ConfMat
                Test_Params['precision_'+str(target_dB)+'dB'] = precision
                Test_Params['recall_'+str(target_dB)+'dB'] = recall
                Test_Params['fscore_'+str(target_dB)+'dB'] = fscore
                Test_Params['PtdLabels_test_'+str(target_dB)+'dB'] = PtdLabels
                Test_Params['Predictions_test_'+str(target_dB)+'dB'] = Predictions
                Test_Params['GroundTruth_test_'+str(target_dB)+'dB'] = GroundTruth
            
            labels = [key for key in PARAMS['classes'].keys()]
            ConfMat_All, precision_All, recall_All, fscore_All = misc.getPerformance(PtdLabels_All, GroundTruths_All, labels)
            Test_Params['ConfMat_All'] = ConfMat_All
            Test_Params['precision_All'] = precision_All
            Test_Params['recall_All'] = recall_All
            Test_Params['fscore_All'] = fscore_All

    return Test_Params




def test_model_generator(PARAMS, Train_Params):
    testingTimeTaken = 0
        
    start = time.process_time()
    if not os.path.exists(PARAMS['opDir']+'/evaluate_generator_results_fold'+str(PARAMS['fold'])+'.pkl'):
        metrics = Train_Params['model'].evaluate(
                generator(PARAMS, PARAMS['folder'], PARAMS['test_files'], PARAMS['batch_size']),
                steps=PARAMS['TS_STEPS'], 
                verbose=1,
                )
        if PARAMS['save_flag']:
            misc.save_obj(metrics, PARAMS['opDir'], 'evaluate_generator_results_fold'+str(PARAMS['fold']))
    else:
        metrics = misc.load_obj(PARAMS['opDir'], 'evaluate_generator_results_fold'+str(PARAMS['fold']))
    
    metrics_names = Train_Params['model'].metrics_names
    
    print(metrics_names)
    print(metrics)
    testingTimeTaken = time.process_time() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    return metrics, metrics_names, testingTimeTaken



def get_train_test_files(cv_file_list, numCV, foldNum):
    train_files = {}
    test_files = {}
    classes = {'music':0, 'speech':1, 'speech+music':2}
    for class_name in classes.keys():
        train_files[class_name] = []
        test_files[class_name] = []
        for i in range(numCV):
            files = cv_file_list[class_name]['fold'+str(i)]
            if foldNum==i:
                test_files[class_name].extend(files)
            else:
                train_files[class_name].extend(files)
    
    return train_files, test_files




def __init__():
    patch_size = 249 # 250ms~24 frames, 500ms~49frames, 750ms~74frames, 1000ms~99frames, 2500ms~249frames
    patch_shift = 24
    opt_n_mels = 120
    opt_l_harm = 21
    opt_l_perc = 11
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            # 'folder': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Scheirer-slaney/',
            # 'folder': '/scratch/mbhattacharjee/data/musan/', # PARAMS ISHAN
            'folder': '/home/phd/mrinmoy.bhattacharjee/data/musan/', # EEE GPU
            # 'folder': '/workspace/pguhap/Mrinmoy/data/musan', # DGX
            # 'feature_folder': './features/', 
            'feature_folder': '/home1/PhD/mrinmoy.bhattacharjee/features/',  # EEE GPU
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'GPU_session':None,
            'TR_STEPS': 0,
            'V_STEPS': 0,
            'test_steps': 0,
            'epochs': 50,
            'batch_size': 16,
            'classes': {0:'music', 1:'speech', 2:'speech_music'},
            'data_augmentation_with_noise': True,
            'Model': 'Lemaire_et_al_MTL', # Lemaire_et_al, Lemaire_et_al_MTL, Lemaire_et_al_Cascaded_MTL, Doukhan_et_al_MTL, Papakostas_et_al_MTL, Jang_et_al_MTL
            'featName': {
                'Lemaire_et_al':'LogMelPercSpec', # LogMelHarmSpec, LogMelPercSpec, LogMelHarmPercSpec 
                'Lemaire_et_al_MTL':'LogMelHarmPercSpec', # LogMelHarmSpec, LogMelPercSpec, LogMelHarmPercSpec 
                'Lemaire_et_al_Cascaded_MTL':'LogMelHarmSpec', # LogMelHarmSpec, LogMelPercSpec, LogMelHarmPercSpec
                'Doukhan_et_al_MTL': 'MelHarmPercSpec',
                'Papakostas_et_al_MTL': 'HarmPercSpec', # 'MelHarmPercSpec',# 'HarmPercSpec',
                'Jang_et_al_MTL': 'LogHarmPercSpec',
                } ,
            'n_fft': {
                'Lemaire_et_al':400, 
                'Lemaire_et_al_MTL':400, 
                'Lemaire_et_al_Cascaded_MTL':400,
                'Doukhan_et_al_MTL': 400,
                'Papakostas_et_al_MTL': 400,
                'Jang_et_al_MTL': 512,
                },
            'n_mels': {
                'Lemaire_et_al':opt_n_mels, 
                'Lemaire_et_al_MTL':opt_n_mels, 
                'Lemaire_et_al_Cascaded_MTL':opt_n_mels,
                'Doukhan_et_al_MTL': opt_n_mels,
                'Papakostas_et_al_MTL': -1,
                'Jang_et_al_MTL': -1,
                },
            'l_harm': {
                'Lemaire_et_al':opt_l_harm, 
                'Lemaire_et_al_MTL':opt_l_harm, 
                'Lemaire_et_al_Cascaded_MTL':opt_l_harm,
                'Doukhan_et_al_MTL': opt_l_harm,
                'Papakostas_et_al_MTL': opt_l_harm,
                'Jang_et_al_MTL': opt_l_harm,
                },
            'l_perc': {
                'Lemaire_et_al':opt_l_perc,
                'Lemaire_et_al_MTL':opt_l_perc,
                'Lemaire_et_al_Cascaded_MTL':opt_l_perc,
                'Doukhan_et_al_MTL': opt_l_perc,
                'Papakostas_et_al_MTL': opt_l_perc,
                'Jang_et_al_MTL': opt_l_perc,
                },
            'input_shape': {
                'Lemaire_et_al':(patch_size,opt_n_mels,1), 
                'Lemaire_et_al_MTL':(patch_size,opt_n_mels,1), 
                'Lemaire_et_al_Cascaded_MTL':(patch_size,opt_n_mels,1),
                'Doukhan_et_al_MTL': (opt_n_mels,patch_size,1),
                'Papakostas_et_al_MTL': (201,patch_size,1),
                'Jang_et_al_MTL': (257,patch_size,1),
                },
            'W':patch_size,
            'W_shift':patch_shift,
            'Tw': 25,
            'Ts': 10,
            'skewness_vector': None, # 'Row', 'Col', 'RowCol', None
            'test_SMR_levels': [-5,0,5,10,15,20],
            'frame_level_scaling': False,
            'dB_wise_test': False,
            'loss_weights': None, # {'S': 0.5, 'M': 0.5, 'R': 0.5, '3C': 1.0}, # None
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
    
    if 'HarmPerc' in PARAMS['featName'][PARAMS['Model']]:
        inp_shape = PARAMS['input_shape'][PARAMS['Model']]
        if 'Lemaire_et_al' in PARAMS['Model']:
            PARAMS['input_shape'][PARAMS['Model']] = (inp_shape[0],2*inp_shape[1],1)
        else:
            PARAMS['input_shape'][PARAMS['Model']] = (2*inp_shape[0],inp_shape[1],1)
        
    PARAMS['feature_opDir'] = PARAMS['feature_folder'] + PARAMS['dataset_name_train'] + '/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])
    
    loss_weights_suffix = ''
    if not PARAMS['loss_weights']==None:
        high_loss = [key for key in PARAMS['loss_weights'].keys()][np.argmax([i for i in PARAMS['loss_weights'].values()])]
        loss_weights_suffix = high_loss + '_high_'
        
    if not PARAMS['skewness_vector']:
        PARAMS['opDir'] = './results/' + PARAMS['dataset_name_train'] + '/Proposed_Work/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '_' + str(len(PARAMS['classes'])) + 'classes_' + loss_weights_suffix + PARAMS['today'] + '/' 
    else:
        PARAMS['opDir'] = './results/' + PARAMS['dataset_name_train'] + '/Proposed_Work/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '_' + PARAMS['skewness_vector'] + 'Skew_' + str(len(PARAMS['classes'])) + 'classes_' + loss_weights_suffix + PARAMS['today'] + '/' 
        if PARAMS['skewness_vector']=='Row':
            inp_shape = PARAMS['input_shape'][PARAMS['Model']]
            PARAMS['input_shape'][PARAMS['Model']] = (1,inp_shape[1],1)
        elif PARAMS['skewness_vector']=='Col':
            inp_shape = PARAMS['input_shape'][PARAMS['Model']]
            PARAMS['input_shape'][PARAMS['Model']] = (inp_shape[0],1,1)

    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
                
    misc.print_configuration(PARAMS)
    
    for PARAMS['fold'] in range(0,1): # range(PARAMS['CV_folds']):
        PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS['cv_file_list'], PARAMS['CV_folds'], PARAMS['fold'])
        
        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()
    
        PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        
        print('input_shape: ', PARAMS['input_shape'][PARAMS['Model']], PARAMS['modelName'])
        Train_Params = perform_training(PARAMS)            
    
        metrics, metrics_names, testingTimeTaken = test_model_generator(PARAMS, Train_Params)
        if len(metrics_names)==0:
            if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
                metrics_names = ['loss', 'S_loss', 'M_loss', 'R_loss', '3C_loss', '3C_accuracy']
            else:
                metrics_names = ['loss', 'accuracy']

        if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
            Test_Params = perform_testing(PARAMS, Train_Params)
            Test_Params['metrics'] = metrics
            Test_Params['metrics_names'] = metrics_names
            Test_Params['testingTimeTaken'] = testingTimeTaken
            if PARAMS['save_flag']:
                misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        else:
            Test_Params = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
            if not 'metrics' in Test_Params.keys():
                Test_Params['metrics'] = metrics
                Test_Params['metrics_names'] = metrics_names
                Test_Params['testingTimeTaken'] = testingTimeTaken
                misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        
        print('Test_Params: ', Test_Params.keys())
        print(Test_Params['precision_annot'], Test_Params['recall_annot'], Test_Params['fscore_annot'])
        
        res_dict = {}
        res_dict['0'] = 'SMR:Annot'
        if ('MTL' in PARAMS['Model']) or ('Cascaded_MTL' in PARAMS['Model']):
            loss_idx = np.squeeze(np.where([name=='loss' for name in metrics_names]))
            print('loss_idx: ', loss_idx)
            res_dict['1'] = 'loss:'+str(metrics[loss_idx])
            acc_idx = np.squeeze(np.where(['3C_acc' in name for name in metrics_names]))
            print('acc_idx: ', acc_idx)
            res_dict['2'] = 'accuracy:'+str(metrics[acc_idx])
        else:
            res_dict['1'] = 'loss:'+str(Test_Params['metrics'][0])
            res_dict['2'] = 'accuracy:'+str(Test_Params['metrics'][1])
        res_dict['3'] = 'Prec_mu:' + str(Test_Params['precision_annot'][0])
        res_dict['4'] = 'Rec_mu:' + str(Test_Params['recall_annot'][0])
        res_dict['5'] = 'F1_mu:' + str(Test_Params['fscore_annot'][0])
        res_dict['6'] = 'Prec_sp:' + str(Test_Params['precision_annot'][1])
        res_dict['7'] = 'Rec_sp:' + str(Test_Params['recall_annot'][1])
        res_dict['8'] = 'F1_sp:' + str(Test_Params['fscore_annot'][1])
        if len(PARAMS['classes'])==3:
            res_dict['9'] = 'Prec_spmu:' + str(Test_Params['precision_annot'][2])
            res_dict['10'] = 'Rec_spmu:' + str(Test_Params['recall_annot'][2])
            res_dict['11'] = 'F1_spmu:' + str(Test_Params['fscore_annot'][2])
        misc.print_results(PARAMS, '', res_dict)
        
        if PARAMS['dB_wise_test']:
            if len(PARAMS['classes'])==3:
                ConfMat_All = Test_Params['ConfMat_All']
                res_dict = {}
                res_dict['0'] = 'SMR:All'
                res_dict['1'] = 'loss:--'
                res_dict['2'] = 'acc:'+str(np.round(np.sum(np.diag(ConfMat_All))/np.sum(ConfMat_All), 4))
                res_dict['3'] = 'Prec_mu:' + str(Test_Params['precision_All'][0])
                res_dict['4'] = 'Rec_mu:' + str(Test_Params['recall_All'][0])
                res_dict['5'] = 'F1_mu:' + str(Test_Params['fscore_All'][0])
                res_dict['6'] = 'Prec_sp:' + str(Test_Params['precision_All'][1])
                res_dict['7'] = 'Rec_sp:' + str(Test_Params['recall_All'][1])
                res_dict['8'] = 'F1_sp:' + str(Test_Params['fscore_All'][1])
                if len(PARAMS['classes'])==3:
                    res_dict['9'] = 'Prec_spmu:'+str(Test_Params['precision_All'][2])
                    res_dict['10'] = 'Rec_spmu:'+str(Test_Params['recall_All'][2])
                    res_dict['11'] = 'F1_spmu:'+str(Test_Params['fscore_All'][2])
                misc.print_results(PARAMS, '', res_dict)
    
                res_dict = {}
                for target_dB in PARAMS['test_SMR_levels']:
                    ConfMat = Test_Params['ConfMat_'+str(target_dB)+'dB']
                    res_dict['0'] = 'SMR:'+str(target_dB)+'dB'
                    res_dict['1'] = 'loss:--'
                    res_dict['2'] = 'acc:'+str(np.round(np.sum(np.diag(ConfMat))/np.sum(ConfMat), 4))
                    res_dict['3'] = 'Prec_mu:' + str(Test_Params['precision_'+str(target_dB)+'dB'][0])
                    res_dict['4'] = 'Rec_mu:' + str(Test_Params['recall_'+str(target_dB)+'dB'][0])
                    res_dict['5'] = 'F1_mu:' + str(Test_Params['fscore_'+str(target_dB)+'dB'][0])
                    res_dict['6'] = 'Prec_sp:' + str(Test_Params['precision_' + str(target_dB)+'dB'][1])
                    res_dict['7'] = 'Rec_sp:' + str(Test_Params['recall_' + str(target_dB)+'dB'][1])
                    res_dict['8'] = 'F1_sp:' + str(Test_Params['fscore_' + str(target_dB)+'dB'][1])
                    if len(PARAMS['classes'])==3:
                        res_dict['9'] = 'Prec_spmu:' + str(Test_Params['precision_'+str(target_dB)+'dB'][2])
                        res_dict['10'] = 'Rec_spmu:' + str(Test_Params['recall_'+str(target_dB)+'dB'][2])
                        res_dict['11'] = 'F1_spmu:' + str(Test_Params['fscore_'+str(target_dB)+'dB'][2])
                          
                    misc.print_results(PARAMS, '', res_dict)
    
        Train_Params = None
        Test_Params = None
    
        if PARAMS['use_GPU']:
            reset_TF_session()
