#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:34:47 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import configparser
import datetime
import numpy as np
import lib.misc as misc
import lib.classifier.classifier_backend as CB
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json



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




def load_model(PARAMS, modelName):
    modelName = '@'.join(modelName.split('.')[:-1]) + '.' + modelName.split('.')[-1]
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
    
    PARAMS['epochs'] = np.load(paramFile)['epochs']
    PARAMS['batch_size'] = np.load(paramFile)['batch_size']
    PARAMS['learning_rate'] = np.load(paramFile)['lr']
    trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
    optimizer = optimizers.Adam(lr=PARAMS['learning_rate'])
    
    with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
        model = model_from_json(f.read())
    model.load_weights(weightFile) # Load weights into the new model
    print(model.summary())
    
    if PARAMS['multi_task_learning']:
        optimizer = optimizers.Nadam(lr=PARAMS['learning_rate'])
        model.compile(
            loss={'SNR':'mean_squared_error', 'S': 'hinge', 'M': 'hinge', '3C':'categorical_crossentropy'}, 
            loss_weights={'SNR':1, 'S':1, 'M':1, '3C':1}, 
            optimizer=optimizer, 
            metrics={'3C':'accuracy'}
            )
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('CNN model exists! Loaded. Training time required=',trainingTimeTaken)
      
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': PARAMS['epochs'],
            'batch_size': PARAMS['batch_size'],
            'learning_rate': PARAMS['learning_rate'],
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params






def perform_testing(PARAMS, Train_Params_H, Train_Params_P):
    PARAMS['HPSS_type'] = 'Harmonic'
    metrics_H, metric_names, testingTimeTaken = CB.test_model_generator(PARAMS, Train_Params_H)
    PARAMS['HPSS_type'] = 'Percussive'
    metrics_P, metric_names, testingTimeTaken = CB.test_model_generator(PARAMS, Train_Params_P)
    metrics = np.add(metrics_H, metrics_P)
    Test_Params = {
        'metrics': metrics,
        'metric_names': metric_names,
        'testingTimeTaken': testingTimeTaken,
        }

    PtdLabels_All = []
    GroundTruths_All = []
    for target_dB in PARAMS['mixing_dB_range']:    
        ConfMat_ensemble, fscore_ensemble, PtdLabels_ensemble, Predictions_ensemble, GroundTruth_ensemble, testingTimeTaken = CB.test_model_ensemble(PARAMS, Train_Params_H, Train_Params_P, target_dB)
        PtdLabels_All.extend(PtdLabels_ensemble)
        GroundTruths_All.extend(GroundTruth_ensemble)
        Test_Params['testingTimeTaken_'+str(target_dB)+'dB'] = testingTimeTaken
        Test_Params['ConfMat_'+str(target_dB)+'dB'] = ConfMat_ensemble
        Test_Params['fscore_'+str(target_dB)+'dB'] = fscore_ensemble
        Test_Params['PtdLabels_test_'+str(target_dB)+'dB'] = PtdLabels_ensemble
        Test_Params['Predictions_test_'+str(target_dB)+'dB'] = Predictions_ensemble
        Test_Params['GroundTruth_test_'+str(target_dB)+'dB'] = GroundTruth_ensemble
        
    ConfMat_All, fscore_All = misc.getPerformance(PtdLabels_All, GroundTruths_All)
    Test_Params['ConfMat_All'] = ConfMat_All
    Test_Params['fscore_All'] = fscore_All

    return Test_Params





'''
Initialize the script
'''
def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Ensemble_Classifier.py']
    PARAMS = {
            'folder': section['folder'], # Folder containing wav files
            'model_folder_H': section['model_folder_H'], # Folder containing fold-wise models trained on MHS
            'model_folder_P': section['model_folder_P'], # Folder containing fold-wise models trained on MPS
            'feature_folder': section['feature_folder'], # Feature Folder
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'CV_folds': int(section['CV_folds']), # Number of cross-validation folds
            'fold': 0,
            'save_flag': section.getboolean('save_flag'),
            'use_GPU': section.getboolean('use_GPU'),
            'train_steps_per_epoch':0,
            'val_steps':0,
            'scale_data': section.getboolean('scale_data'),
            'PCA_flag': section.getboolean('PCA_flag'),
            'GPU_session':None,
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'epochs': int(section['epochs']),
            'batch_size': int(section['batch_size']),
            'Tw': int(section['Tw']),
            'Ts': int(section['Ts']),
            'n_mels': int(section['n_mels']),
            'silThresh': float(section['silThresh']),
            'multi_task_learning': section.getboolean('multi_task_learning'),
            'CNN_architecture_name': section['CNN_architecture_name'], # Papakostas_et_al, Doukhan_et_al
            'output_folder':'',
            'classes': {0:'music', 1:'speech'},
            'mixing_dB_range': [-5, -2, -1, 0, 2, 5, 8, 10, 20],
            'featName': 'MelHPSS',
            'HPSS_type': '', # Harmonic, Percussive, Both
            'opDir':'',
            'modelName':'',
            'task': 'Classification',
            }

    if not os.path.exists(PARAMS['folder']+'/Dataset_Duration.pkl'):
        PARAMS['total_duration'], PARAMS['filewise_duration'] = misc.calculate_dataset_size(PARAMS)
        if PARAMS['save_flag']:
            misc.save_obj({'total_duration':PARAMS['total_duration'], 'filewise_duration':PARAMS['filewise_duration']}, PARAMS['folder'], 'Dataset_Duration')
    else:
        PARAMS['total_duration'] = misc.load_obj(PARAMS['folder'], 'Dataset_Duration')['total_duration']
        PARAMS['filewise_duration'] = misc.load_obj(PARAMS['folder'], 'Dataset_Duration')['filewise_duration']

    total_duration = []
    for classname in PARAMS['total_duration']:
        total_duration.append(PARAMS['total_duration'][classname])
    PARAMS['dataset_size'] = int(3*np.min(total_duration)/3600) # Hours
    print('Dataset size: ', PARAMS['dataset_size'], 'Hrs', total_duration)

            
    interval_shift = PARAMS['CNN_patch_shift']*PARAMS['Ts'] # Frame shift in milisecs
    PARAMS['train_steps_per_epoch'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.7/(2*PARAMS['batch_size']))
    PARAMS['val_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.3/(2*PARAMS['batch_size']))
    PARAMS['test_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.33/(2*PARAMS['batch_size']))
    print('train_steps_per_epoch: %d, \tval_steps: %d,  \ttest_steps: %d\n'%(PARAMS['train_steps_per_epoch'], PARAMS['val_steps'], PARAMS['test_steps']))
   
    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()

    '''
    Initializations
    ''' 
    PARAMS['input_shape'] = (PARAMS['n_mels'], PARAMS['CNN_patch_size'], 1)

    if PARAMS['multi_task_learning']:
        opDir_suffix = '_CNN_MTL/'
    else:
        opDir_suffix = '_CNN/'
    PARAMS['feature_opDir'] = PARAMS['feature_folder'] + '/' + PARAMS['folder'].split('/')[-2] + '/MelHPSS/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])

    cv_file_list = misc.create_CV_folds(PARAMS, PARAMS['folder'], PARAMS['classes'], PARAMS['CV_folds'])
    cv_file_list_test = cv_file_list
    PARAMS['test_folder'] = PARAMS['folder']
    PARAMS['opDir'] = PARAMS['feature_opDir'] + '/__RESULTS/' + PARAMS['today'] + '/Harmonic_Percussive_LF' + opDir_suffix
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
                
    misc.print_configuration(PARAMS)
                
    for foldNum in range(PARAMS['CV_folds']):
        PARAMS['fold'] = foldNum
        PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
        
        if PARAMS['use_GPU']:
            PARAMS['GPU_session'] = start_GPU_session()

        PARAMS['modelName_H'] = PARAMS['model_folder_H'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        Train_Params_H = load_model(PARAMS, PARAMS['modelName_H'])
        PARAMS['modelName_P'] = PARAMS['model_folder_P'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        Train_Params_P = load_model(PARAMS, PARAMS['modelName_P'])
        
        print('input_shape: ', PARAMS['input_shape'], PARAMS['modelName_H'], PARAMS['modelName_P'])

        if not os.path.exists(PARAMS['opDir']+'/Test_Params_fold'+str(PARAMS['fold'])+'.pkl'):
            Test_Params = perform_testing(PARAMS, Train_Params_H, Train_Params_P)
            if PARAMS['save_flag']:
                misc.save_obj(Test_Params, PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        else:
            Test_Params = misc.load_obj(PARAMS['opDir'], 'Test_Params_fold'+str(PARAMS['fold']))
        print('Test accuracy=', Test_Params['fscore_20dB'])

        res_dict = {}
        if not PARAMS['multi_task_learning']:
            res_dict['0'] = Test_Params['metric_names'][0]+':'+str(Test_Params['metrics'][0])
            res_dict['1'] = Test_Params['metric_names'][1]+':'+str(Test_Params['metrics'][1])
            ln = 2
            
        else:
            Test_Params['metric_names'] = ['loss', 'NO_loss', 'MU_loss', 'HG_loss', '3C_loss', '3C_accuracy']
            for i in range(len(Test_Params['metric_names'])):
                res_dict[str(i)] = Test_Params['metric_names'][i]+':'+str(np.round(Test_Params['metrics'][i], 4))
            ln = 5

        res_dict[str(ln)] = 'All_F1_mu:' + str(Test_Params['fscore_All'][0])
        res_dict[str(ln+1)] = 'All_F1_sp:' + str(Test_Params['fscore_All'][1])
        res_dict[str(ln+2)] = 'All_F1_spmu:' + str(Test_Params['fscore_All'][2])
        res_dict[str(ln+3)] = 'All_F1_avg:' + str(Test_Params['fscore_All'][3])
        ln += 4

        for target_dB in PARAMS['mixing_dB_range']:
            res_dict[str(ln)] = str(target_dB)+'dB_F1_mu:' + str(Test_Params['fscore_'+str(target_dB)+'dB'][0])
            res_dict[str(ln+1)] = str(target_dB)+'dB_F1_sp:' + str(Test_Params['fscore_' + str(target_dB)+'dB'][1])
            res_dict[str(ln+2)] = str(target_dB)+'dB_F1_spmu:' + str(Test_Params['fscore_' + str(target_dB)+'dB'][2])
            res_dict[str(ln+3)] = str(target_dB)+'dB_F1_avg:' + str(Test_Params['fscore_' + str(target_dB)+'dB'][3])
            ln += 4
                        
        misc.print_results(PARAMS, '', res_dict)
                    
        Train_Params = None
        Test_Params = None

        if PARAMS['use_GPU']:
            reset_TF_session()

