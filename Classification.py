#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:51:23 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import datetime
import lib.misc as misc
import configparser
import lib.classifier.classifier_backend as CB
import numpy as np
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




def perform_training(PARAMS):
    PARAMS['modelName'] = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = PARAMS['modelName'].split('.')[0] + '.h5'
    architechtureFile = PARAMS['modelName'].split('.')[0] + '.json'
    paramFile = PARAMS['modelName'].split('.')[0] + '_params.npz'
    logFile = PARAMS['modelName'].split('.')[0] + '_log.csv'

    PARAMS['modelName'] = '.'.join(PARAMS['modelName'].split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    
    print('paramFile: ', paramFile)
    
    if not os.path.exists(paramFile):
        if PARAMS['initialize_CNN_model'] and PARAMS['multi_task_learning']:
            print('MTL model initialized with trained CNN model!')

            trained_model_weightFile = PARAMS['model_folder'] + '/fold' + str(PARAMS['fold']) + '_model.h5'
            PARAMS_temp = PARAMS.copy()
            PARAMS_temp['multi_task_learning'] = False
            model, PARAMS['learning_rate'] = CB.get_cnn_model(PARAMS_temp)

            model.load_weights(trained_model_weightFile) # Load weights into the new model
            
            model = CB.get_MTL_converted_model(PARAMS, model)            
           
        else:
            model, PARAMS['learning_rate'] = CB.get_cnn_model(PARAMS)

        model, trainingTimeTaken, History = CB.train_model(PARAMS, model, weightFile)
        
        if PARAMS['save_flag']:
            model.save_weights(weightFile) # Save the weights
            with open(architechtureFile, 'w') as f: # Save the model architecture
                f.write(model.to_json())
            np.savez(paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], lr=PARAMS['learning_rate'], trainingTimeTaken=trainingTimeTaken)
        print('CNN model trained.')
    else:
        PARAMS['epochs'] = np.load(paramFile)['epochs']
        PARAMS['batch_size'] = np.load(paramFile)['batch_size']
        PARAMS['learning_rate'] = np.load(paramFile)['lr']
        trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        optimizer = optimizers.Adam(lr=PARAMS['learning_rate'])
        
        with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
            model = model_from_json(f.read())
        model.load_weights(weightFile) # Load weights into the new model
        
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




def perform_testing(PARAMS, Train_Params):
    metrics, metric_names, testingTimeTaken = CB.test_model_generator(PARAMS, Train_Params)
    Test_Params = {
        'metrics': metrics,
        'metric_names': metric_names,
        'testingTimeTaken': testingTimeTaken,
        }

    PtdLabels_All = []
    GroundTruths_All = []
    for target_dB in PARAMS['mixing_dB_range']:    
        ConfMat, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken, fscore_mu, fscore_sp, MSE_SMR = CB.test_model(PARAMS, Train_Params, target_dB)
        PtdLabels_All.extend(PtdLabels)
        GroundTruths_All.extend(GroundTruth)
        Test_Params['testingTimeTaken_'+str(target_dB)+'dB'] = testingTimeTaken
        Test_Params['ConfMat_'+str(target_dB)+'dB'] = ConfMat
        Test_Params['fscore_'+str(target_dB)+'dB'] = fscore
        Test_Params['fscore_mu_'+str(target_dB)+'dB'] = fscore_mu
        Test_Params['fscore_sp_'+str(target_dB)+'dB'] = fscore_sp
        Test_Params['MSE_SMR_'+str(target_dB)+'dB'] = MSE_SMR
        Test_Params['PtdLabels_test_'+str(target_dB)+'dB'] = PtdLabels
        Test_Params['Predictions_test_'+str(target_dB)+'dB'] = Predictions
        Test_Params['GroundTruth_test_'+str(target_dB)+'dB'] = GroundTruth
        
    ConfMat_All, fscore_All = misc.getPerformance(PtdLabels_All, GroundTruths_All)
    Test_Params['ConfMat_All'] = ConfMat_All
    Test_Params['fscore_All'] = fscore_All

    return Test_Params




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Classification.py']
    PARAMS = {
            'folder': section['folder'], # Folder containing wav files
            'feature_folder': section['feature_folder'], # Folder containing features files
            'test_path': section['test_path'], # Test features path, empty if same as feature_folder
            'model_folder': section['model_folder'], # Path to existing model files, empty otherwise
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'CV_folds': int(section['CV_folds']), # Number of cross-validation folds
            'fold': 0,
            'save_flag': section.getboolean('save_flag'),
            'use_GPU': section.getboolean('use_GPU'),
            'train_steps_per_epoch': 0,
            'val_steps': 0,
            'test_steps': 0,
            'scale_data': section.getboolean('scale_data'),
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'epochs': int(section['epochs']),
            'batch_size': int(section['batch_size']),
            'Tw': int(section['Tw']),
            'Ts': int(section['Ts']),
            'feature_type': ['Melspectrogram'], # [Melspectrogram', 'MelHPSS'],
            'HPSS_type': section['HPSS_type'], # Harmonic, Percussive, Both
            'n_mels': int(section['n_mels']),
            'silThresh': float(section['silThresh']),
            'multi_task_learning': section.getboolean('multi_task_learning'),
            'initialize_CNN_model': section.getboolean('initialize_CNN_model'),
            'freeze_CNN': section.getboolean('freeze_CNN'),
            'CNN_architecture_name': section['CNN_architecture_name'], # Papakostas_et_al, Doukhan_et_al
            'GPU_session':None,
            'output_folder':'',
            'classes': {0:'music', 1:'speech'},
            'mixing_dB_range': [-5, -2, -1, 0, 2, 5, 8, 10, 20],
            'dataset':'',
            'opDir':'',
            'modelName':'',
            'input_dim':0,
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


    for PARAMS['featName'] in PARAMS['feature_type']:

        '''
        Initializations
        ''' 
        if PARAMS['featName']=='Melspectrogram':
            PARAMS['input_shape'] = (PARAMS['n_mels'], PARAMS['CNN_patch_size'], 1)
        elif PARAMS['featName']=='MelHPSS':
            if PARAMS['HPSS_type']=='Both':
                PARAMS['input_shape'] = (PARAMS['n_mels'], PARAMS['CNN_patch_size'], 2)
            else:
                PARAMS['input_shape'] = (PARAMS['n_mels'], PARAMS['CNN_patch_size'], 1)

        PARAMS['feature_opDir'] = PARAMS['feature_folder'] + '/' + PARAMS['folder'].split('/')[-2] + '/' + PARAMS['featName'] + '/'            
        if not os.path.exists(PARAMS['feature_opDir']):
            os.makedirs(PARAMS['feature_opDir'])

        if PARAMS['multi_task_learning']:
            opDir_suffix = '_MTL_' + PARAMS['CNN_architecture_name']
        else:
            opDir_suffix = '_' + PARAMS['CNN_architecture_name']
            
        if PARAMS['test_path']=='':
            cv_file_list = misc.create_CV_folds(PARAMS, PARAMS['folder'], PARAMS['classes'], PARAMS['CV_folds'])
            cv_file_list_test = cv_file_list
            PARAMS['test_folder'] = PARAMS['folder']
            PARAMS['output_folder'] = PARAMS['feature_opDir'] + '/__RESULTS/' + PARAMS['today'] + '/'
        else:
            cv_file_list = misc.create_CV_folds(PARAMS, PARAMS['folder'], PARAMS['classes'], PARAMS['CV_folds'])
            cv_file_list_test = misc.create_CV_folds(PARAMS, PARAMS['feature_opDir'], PARAMS['classes'], PARAMS['CV_folds'])
            PARAMS['test_folder'] = PARAMS['test_path']
            PARAMS['output_folder'] = PARAMS['feature_opDir'] + '/__RESULTS/' + PARAMS['today'] + '/'
            opDir_suffix += '_GEN_PERF_' + PARAMS['folder'].split('/')[-2]

        if PARAMS['featName']=='MelHPSS':
            PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['featName'] + '_' + PARAMS['HPSS_type'] + '_CNN' + opDir_suffix + '/'
        else:
            PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['featName'] + '_CNN' + opDir_suffix + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
                    
        misc.print_configuration(PARAMS)
                    
        for foldNum in range(PARAMS['CV_folds']):
            PARAMS['fold'] = foldNum
            PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
            
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

            if PARAMS['multi_task_learning']:
                res_dict = {}
                ln = 0
                for target_dB in PARAMS['mixing_dB_range']:
                    res_dict[str(ln)] = 'F1_sp_'+str(target_dB)+'dB:' + str(Test_Params['fscore_sp_'+str(target_dB)+'dB'][0])
                    res_dict[str(ln+1)] = 'F1_nonsp_'+str(target_dB)+'dB:' + str(Test_Params['fscore_sp_'+str(target_dB)+'dB'][1])
                    res_dict[str(ln+2)] = 'F1_avg_'+str(target_dB)+'dB:' + str(Test_Params['fscore_sp_'+str(target_dB)+'dB'][2])
                    ln += 3
                    res_dict[str(ln)] = 'F1_mu_'+str(target_dB)+'dB:' + str(Test_Params['fscore_mu_'+str(target_dB)+'dB'][0])
                    res_dict[str(ln+1)] = 'F1_nonmu_'+str(target_dB)+'dB:' + str(Test_Params['fscore_mu_'+str(target_dB)+'dB'][1])
                    res_dict[str(ln+2)] = 'F1_avg_'+str(target_dB)+'dB:' + str(Test_Params['fscore_mu_'+str(target_dB)+'dB'][2])
                    ln += 3
                    res_dict[str(ln)] = 'MSE_SMR_'+str(target_dB)+'dB:' + str(Test_Params['MSE_SMR_'+str(target_dB)+'dB'])
                    ln += 1
                misc.print_results(PARAMS, 'Auxiliary_tasks', res_dict)
                    
            Train_Params = None
            Test_Params = None
    
            if PARAMS['use_GPU']:
                reset_TF_session()
