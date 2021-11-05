#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:44:47 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import sys
import datetime
import lib.misc as misc
from lib.baseline_architectures import get_Lemaire_model
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
import time
import lib.preprocessing as preproc
from lib.cython_impl.tools import scale_data as cscale_data




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
            



def train_model(PARAMS, model, weightFile, logFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    # logFile = '/'.join(weightFile.split('/')[:-2]) + '/log_fold' + str(PARAMS['fold']) + '.csv'
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
        print('n_mels: ', PARAMS['n_mels'][PARAMS['Model']])
        model, learning_rate = get_Lemaire_model(TR_STEPS=PARAMS['TR_STEPS'],  N_MELS=PARAMS['n_mels'][PARAMS['Model']], n_classes=len(PARAMS['classes']))

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
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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
    metric_names = Train_Params['model'].metrics_names
    
    print(metric_names)
    print(metrics)
    testingTimeTaken = time.process_time() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    return metrics, metric_names, testingTimeTaken




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
    if PARAMS['Model']=='Lemaire_et_al':
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
            fl_count = 0
            for fl in files:
                fl_count += 1
                fName = PARAMS['folder'] + '/'+ classname + '/' + fl
                if not os.path.exists(fName):
                    continue
                if classname=='speech':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, fName, '', None)
                    pred = Train_Params['model'].predict(x=batchData)
                elif classname=='music':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, '', fName, None)
                    pred = Train_Params['model'].predict(x=batchData)
                pred_lab = np.argmax(pred, axis=1)
                if np.size(Predictions)<=1:
                    Predictions = pred
                    PtdLabels = np.argmax(pred, axis=1)
                    if classname=='speech':
                        GroundTruth = np.array([1]*np.shape(pred)[0])
                    elif classname=='music':
                        GroundTruth = np.array([0]*np.shape(pred)[0])
                else:
                    Predictions = np.append(Predictions, pred, 0)
                    PtdLabels = np.append(PtdLabels, np.argmax(pred, axis=1))
                    if classname=='speech':
                        GroundTruth = np.append(GroundTruth, np.array([1]*np.shape(pred)[0]))
                    elif classname=='music':
                        GroundTruth = np.append(GroundTruth, np.array([0]*np.shape(pred)[0]))
                    
                print(fl_count, '/', len(files), target_dB, 'dB\t', classname, 'pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), end='\t', flush=True)
                if classname=='speech':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==1)*100/len(pred_lab), 4)
                    acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
                    print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', acc_fl, acc_all)
                elif classname=='music':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==0)*100/len(pred_lab), 4)
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
            pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)
            if np.size(Predictions)<=1:
                Predictions = pred
                GroundTruth = np.array([2]*np.shape(pred)[0])
                PtdLabels = np.argmax(pred, axis=1)
            else:
                Predictions = np.append(Predictions, pred, 0)
                GroundTruth = np.append(GroundTruth, np.array([2]*np.shape(pred)[0]))
                PtdLabels = np.append(PtdLabels, np.argmax(pred, axis=1))
            acc_fl = np.round(np.sum(np.array(pred_lab)==2)*100/len(pred_lab), 4)
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
    metrics, metric_names, testingTimeTaken = test_model_generator(PARAMS, Train_Params)
    Test_Params = {
        'metrics': metrics,
        'metric_names': metric_names,
        'testingTimeTaken': testingTimeTaken,
        }

    ConfMat, precision, recall, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken = test_model(PARAMS, Train_Params, None)
    Test_Params['testingTimeTaken_annot'] = testingTimeTaken
    Test_Params['ConfMat_annot'] = ConfMat
    Test_Params['precision_annot'] = precision
    Test_Params['recall_annot'] = recall
    Test_Params['fscore_annot'] = fscore
    Test_Params['PtdLabels_test_annot'] = PtdLabels
    Test_Params['Predictions_test_annot'] = Predictions
    Test_Params['GroundTruth_test_annot'] = GroundTruth

    return Test_Params





def get_train_test_files(cv_file_list, tuning_fold=0):
    '''
    Getting files only from the tuning fold for hyperparameter tuning.

    Parameters
    ----------
    cv_file_list : dict
        Distribution of files from all classes into 3 cross-validation folds.
    tuning_fold : int, optional
        The cross-valdiation fold to use for parameter tuning. The default is 0.

    Returns
    -------
    train_files : dict
        Class-wise training files.
    test_files : dict
        Class-wise testing files.

    '''
    train_files = {}
    test_files = {}
    classes = {'music':0, 'speech':1, 'speech+music':2}
    for class_name in classes.keys():
        files = cv_file_list[class_name]['fold'+str(tuning_fold)]
        nTrain = int(len(files)*0.7)
        train_files[class_name] = files[:nTrain]
        test_files[class_name] = files[nTrain:]
    
    return train_files, test_files



def __init__():
    '''
    Configuring the experimental setup.

    Returns
    -------
    PARAMS : Dict
        Configuration information.

    '''
    patch_size = 68
    patch_shift = 68
    opt_n_mel = 120
    opt_l_harm = 21
    opt_l_perc = 11
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            # 'folder': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Scheirer-slaney/',
            'folder': '/scratch/mbhattacharjee/data/musan/', # PARAMS ISHAN
            # 'folder': '/home/phd/mrinmoy.bhattacharjee/data/musan/', # EEE GPU
            # 'folder': '/workspace/pguhap/Mrinmoy/data/musan', # DGX
            'feature_folder': './features/', 
            # 'feature_folder': '/home1/PhD/mrinmoy.bhattacharjee/features/',  # EEE GPU
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
            'classes': {0:'music', 1:'speech', 2:'speech_music'}, # {0:'music', 1:'speech', 2:'speech_music'}
            'data_augmentation_with_noise': True,
            'W':patch_size,
            'W_shift':patch_shift,
            'Tw': 25,
            'Ts': 10,
            'n_fft': {'Lemaire_et_al':400},
            'n_mels': {'Lemaire_et_al':opt_n_mel}, # original 80, optimized to 20
            'input_shape': {'Lemaire_et_al':(patch_size,opt_n_mel,1)},
            'l_harm': {'Lemaire_et_al':opt_l_harm},
            'l_perc': {'Lemaire_et_al':opt_l_perc},
            'featName': {'Lemaire_et_al':'LogMelSpec'} , # {'Lemaire_et_al':'LogMelSpec'}, {'Lemaire_et_al':'LogMelHarmSpec'}, {'Lemaire_et_al':'LogMelPercSpec'}
            'tuning_param': 'l_perc', # 'n_mels', 'l_harm', 'l_perc', 'W', 'loss_weights'
            'param_ranges':{
                'n_mels':[20, 40, 60, 80, 100, 120],
                'l_harm': [11, 21, 31, 41, 51],
                'l_perc': [11, 21, 31, 41, 51],
                'W': [25, 50, 75, 100],
                'loss_weights': [
                    {'3C_high':[0.2,0.2,0.2,0.4]}, 
                    {'SMR_high':[0.2,0.2,0.4,0.2]}, 
                    {'M_high':[0.2,0.4,0.2,0.2]}, 
                    {'S_high':[0.4,0.2,0.2,0.2]},
                    ],
                },
            'test_SMR_levels': [-5,0,5,10,15,20],
            'frame_level_scaling': False,
            'nGPU': 1,
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
        DT_SZ += PARAMS['cv_file_list']['total_duration'][classname]/3 # in Hours
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
        
    for param_val in PARAMS['param_ranges'][PARAMS['tuning_param']]:
        PARAMS[PARAMS['tuning_param']][PARAMS['Model']] = param_val
        if PARAMS['tuning_param']=='n_mels':
            PARAMS['input_shape'][PARAMS['Model']] = (PARAMS['W'], param_val, 1)

        PARAMS['feature_opDir'] = PARAMS['feature_folder'] + PARAMS['dataset_name_train'] + '/' + PARAMS['Model'] + '_Hyperparameter_Selection/' + PARAMS['tuning_param'] + '/' + PARAMS['featName'][PARAMS['Model']] + '_' + str(param_val) + '/'
        if not os.path.exists(PARAMS['feature_opDir']):
            os.makedirs(PARAMS['feature_opDir'])
            
        PARAMS['opDir'] = './results/' + PARAMS['dataset_name_train'] + '/Hyperparameter_Selection/' +  PARAMS['tuning_param'] + '_' + PARAMS['today'] + '/'  + str(param_val) + '_' + PARAMS['featName'][PARAMS['Model']] + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
                    
        misc.print_configuration(PARAMS)
        
        for PARAMS['fold'] in range(0,1): # range(PARAMS['CV_folds']):
            PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS['cv_file_list'], PARAMS['fold'])
    
            if PARAMS['use_GPU']:
                PARAMS['GPU_session'] = start_GPU_session()
        
            PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
            
            print('input_shape: ', PARAMS['input_shape'], PARAMS['modelName'])
            Train_Params = perform_training(PARAMS)            
        
            if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
                Test_Params = perform_testing(PARAMS, Train_Params)
                if PARAMS['save_flag']:
                    misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
            else:
                Test_Params = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
    
            print('Test_Params: ', Test_Params.keys())
            print(Test_Params['precision_annot'], Test_Params['recall_annot'], Test_Params['fscore_annot'])
            
            res_dict = {}
            res_dict['0'] = 'SMR:Annot'
            res_dict['1'] = Test_Params['metric_names'][0]+':'+str(Test_Params['metrics'][0])
            res_dict['2'] = Test_Params['metric_names'][1]+':'+str(Test_Params['metrics'][1])
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
                        
            Train_Params = None
            Test_Params = None
        
            if PARAMS['use_GPU']:
                reset_TF_session()
