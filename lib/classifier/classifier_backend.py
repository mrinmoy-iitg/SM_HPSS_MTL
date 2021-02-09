#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:00:47 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input, Concatenate, MaxPooling2D, Activation, Dense, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_uniform, RandomNormal, Constant

import time
import lib.misc as misc
import lib.feature.preprocessing as preproc




def generator(PARAMS, folder, file_list, batchSize):
    batch_count = 0
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])

    file_list_sp_temp = file_list['speech'].copy()
    file_list_mu_temp = file_list['music'].copy()
    file_list_spmu_sp = file_list['speech'].copy()
    file_list_spmu_mu = file_list['music'].copy()

    batchData_sp = np.empty([], dtype=float)
    batchData_mu = np.empty([], dtype=float)
    batchData_spmu = np.empty([], dtype=float)
    balance_sp = 0
    balance_mu = 0
    balance_spmu = 0

    if not os.path.exists(PARAMS['feature_opDir']+'/speech/'):
        os.makedirs(PARAMS['feature_opDir']+'/speech/')
        os.makedirs(PARAMS['feature_opDir']+'/music/')
        os.makedirs(PARAMS['feature_opDir']+'/speech_music/')
        
    while 1:
        batchData = np.empty([], dtype=float)
        batchLabel1 = np.empty([], dtype=float)
        batchLabel2 = np.empty([], dtype=float)
        batchLabel3 = np.empty([], dtype=float)
        target_dB = np.random.choice(PARAMS['mixing_dB_range'])
        
        while balance_sp<batchSize:
            if not file_list_sp_temp:
                file_list_sp_temp = file_list['speech'].copy()
            sp_fName = file_list_sp_temp.pop()
            sp_fName_path = folder + '/speech/' + sp_fName
            if not os.path.exists(sp_fName_path):
                continue
            file_list_spmu_sp.append(sp_fName)         
            fv_sp = preproc.get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', target_dB)
            fv_sp_patches = preproc.get_feature_patches(PARAMS, fv_sp, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['featName'])
            if balance_sp==0:
                batchData_sp = fv_sp_patches
            else:
                batchData_sp = np.append(batchData_sp, fv_sp_patches, axis=0)
            balance_sp += np.shape(fv_sp_patches)[0]
            

        while balance_mu<batchSize:
            if not file_list_mu_temp:
                file_list_mu_temp = file_list['music'].copy()
            mu_fName = file_list_mu_temp.pop()
            mu_fName_path = folder + '/music/' + mu_fName
            if not os.path.exists(mu_fName_path):
                continue
            file_list_spmu_mu.append(mu_fName)
            fv_mu = preproc.get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, target_dB)
            fv_mu_patches = preproc.get_feature_patches(PARAMS, fv_mu, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['featName'])
            if balance_mu==0:
                batchData_mu = fv_mu_patches
            else:
                batchData_mu = np.append(batchData_mu, fv_mu_patches, axis=0)
            balance_mu += np.shape(fv_mu_patches)[0]


        while balance_spmu<batchSize:
            if not file_list_spmu_sp:
                file_list_spmu_sp = file_list['speech'].copy()
            np.random.shuffle(file_list_spmu_sp)
            sp_fName = file_list_spmu_sp.pop()
            sp_fName_path = folder + '/speech/' + sp_fName

            if not file_list_spmu_mu:
                file_list_spmu_mu = file_list['music'].copy()
            np.random.shuffle(file_list_spmu_mu)
            mu_fName = file_list_spmu_mu.pop()
            mu_fName_path = folder + '/music/' + mu_fName
            fv_spmu = preproc.get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], sp_fName_path, mu_fName_path, target_dB)
            fv_spmu_patches = preproc.get_feature_patches(PARAMS, fv_spmu, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['featName'])
            if balance_spmu==0:
                batchData_spmu = fv_spmu_patches
            else:
                batchData_spmu = np.append(batchData_spmu, fv_spmu_patches, axis=0)
            balance_spmu += np.shape(fv_spmu_patches)[0]

        batchData = batchData_sp[:batchSize, :] # speech label=1
        batchData = np.append(batchData, batchData_mu[:batchSize, :], axis=0) # music label=0
        batchData = np.append(batchData, batchData_spmu[:batchSize, :], axis=0)  # speech_music label=2
        
        balance_sp -= batchSize
        balance_mu -= batchSize
        balance_spmu -= batchSize
        batchData_sp = batchData_sp[batchSize:, :]
        batchData_mu = batchData_mu[batchSize:, :]            
        batchData_spmu = batchData_spmu[batchSize:, :]            

        if not PARAMS['multi_task_learning']:
            class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
            batchLabel = np.ones(3*batchSize)
            batchLabel[:batchSize] = class_labels['speech']
            batchLabel[batchSize:2*batchSize] = class_labels['music']
            batchLabel[2*batchSize:] = 2
            OHE_batchLabel = to_categorical(batchLabel, num_classes=3)
        
            batch_count += 1
            # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(OHE_batchLabel))
            yield batchData, OHE_batchLabel
            
        else:
            '''
            Pure Impure
            '''
            batchLabel1 = np.ones((3*batchSize,2))
            batchLabel1[:batchSize, :] = np.repeat(np.array([0, 1], ndmin=2), batchSize, axis=0)
            batchLabel1[batchSize:2*batchSize] = np.repeat(np.array([1, 0], ndmin=2), batchSize, axis=0)
            if target_dB>=0:
                batchLabel1[2*batchSize:] = np.repeat(np.array([1/np.power(10,(target_dB/10)), 1], ndmin=2), batchSize, axis=0)
            else:
                batchLabel1[2*batchSize:] = np.repeat(np.array([1, np.power(10,(target_dB/10))], ndmin=2), batchSize, axis=0)                
    
            '''
            Speech Nonspeech
            '''
            batchLabel2 = np.ones(3*batchSize)
            batchLabel2[:batchSize] = 1
            batchLabel2[batchSize:2*batchSize] = -1
            batchLabel2[2*batchSize:] = -1
    
            '''
            Music Nonmusic
            '''
            batchLabel3 = np.ones(3*batchSize)
            batchLabel3[:batchSize] = -1
            batchLabel3[batchSize:2*batchSize] = 1
            batchLabel3[2*batchSize:] = -1

            '''
            Multi-class
            '''
            batchLabel4 = np.ones(3*batchSize)
            batchLabel4[:batchSize] = 1
            batchLabel4[batchSize:2*batchSize] = 0
            batchLabel4[2*batchSize:] = 2
            OHE_batchLabel4 = to_categorical(batchLabel4, num_classes=3)            
            
            batchLabel = {'SNR': batchLabel1, 'S': batchLabel2, 'M': batchLabel3, '3C': OHE_batchLabel4}
    
            batch_count += 1
            yield batchData, batchLabel




def train_model(PARAMS, model, weightFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    logFile = '/'.join(weightFile.split('/')[:-2]) + '/log_fold' + str(PARAMS['fold']) + '.csv'
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.clock()

    SPE = PARAMS['train_steps_per_epoch']
    SPE_val = PARAMS['val_steps']
    print('SPE: ', SPE, SPE_val)
    
    train_files = {}
    val_files = {}
    for classname in  PARAMS['train_files'].keys():
        files = PARAMS['train_files'][classname]
        np.random.shuffle(files)
        idx = int(len(files)*0.7)
        train_files[classname] = files[:idx]
        val_files[classname] = files[idx:]
    
    # Train the model
    History = model.fit(
            generator(PARAMS, PARAMS['folder'], train_files, PARAMS['batch_size']),
            steps_per_epoch = SPE,
            validation_data = generator(PARAMS, PARAMS['folder'], val_files, PARAMS['batch_size']), 
            validation_steps = SPE_val,
            epochs=PARAMS['epochs'], 
            verbose=1,
            callbacks=[csv_logger, es, mcp],
            # shuffle=True,
            )

    trainingTimeTaken = time.clock() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History



def get_MTL_converted_model(PARAMS, source_model):
    
    optimizer = optimizers.Adam(lr=PARAMS['learning_rate'])
    source_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])        
    print(source_model.summary())
    
    if PARAMS['freeze_CNN']:
        source_model.trainable = False # Freezing the model

    x = source_model.layers[-6].output

    x1 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0), name='MTL_D1')(x)
    x1 = BatchNormalization(name='MTL_BN1')(x1)
    x1 = Activation('relu', name='MTL_Act1')(x1)
    x1 = Dropout(0.5, name='MTL_DO1')(x1)
    output1 = Dense(2, activation='linear', kernel_regularizer=l2(), name='SNR')(x1)

    x2 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0), name='MTL_D2')(x)
    x2 = BatchNormalization(name='MTL_BN2')(x2)
    x2 = Activation('relu', name='MTL_Act2')(x2)
    x2 = Dropout(0.5, name='MTL_DO2')(x2)
    output2 = Dense(1, activation='tanh', kernel_regularizer=l2(), name='S')(x2)

    x3 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0), name='MTL_D3')(x)
    x3 = BatchNormalization(name='MTL_BN3')(x3)
    x3 = Activation('relu', name='MTL_Act3')(x3)
    x3 = Dropout(0.5, name='MTL_DO3')(x3)
    output3 = Dense(1, activation='tanh', kernel_regularizer=l2(), name='M')(x3)
    
    x4 = Concatenate(axis=-1)([output1, output2, output3])
    x4 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0), name='MTL_D4')(x4)
    x4 = BatchNormalization(name='MTL_BN4')(x4)
    x4 = Activation('relu', name='MTL_Act4')(x4)
    x4 = Dropout(0.5, name='MTL_DO4')(x4)
    output4 = Dense(3, activation='softmax', kernel_regularizer=l2(), name='3C')(x4)

    model = Model(source_model.layers[0].input, [output1, output2, output3, output4])
    learning_rate = 0.0001
    optimizer = optimizers.Nadam(lr=learning_rate)
    model.compile(
        loss={'SNR':'mean_squared_error', 'S': 'hinge', 'M': 'hinge', '3C':'categorical_crossentropy'}, 
        loss_weights={'SNR':1, 'S':1, 'M':1, '3C':1}, 
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )

    print('Multi-task learning architecture (adapted from CNN model of Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION)\n', model.summary())

    return model






def get_cnn_model(PARAMS):
    
    if PARAMS['CNN_architecture_name'] == 'Doukhan_et_al':
        '''
        Baseline :- Doukhan et. al. MIREX 2018 MUSIC AND SPEECH DETECTION SYSTEM
        '''
        input_img = Input(PARAMS['input_shape'])
    
        x = Conv2D(64, input_shape=PARAMS['input_shape'], kernel_size=(4, 5), strides=(1, 1), kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1, 12), strides=(1, 12))(x)
        
        x = Flatten()(x)
    
        x = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
    
        x = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
    
        x = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        if not PARAMS['multi_task_learning']:
            x = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
    
            output = Dense(len(PARAMS['classes'])+1, activation='softmax', kernel_regularizer=l2())(x)
        
            model = Model(input_img, output)
            learning_rate = 0.0001
            optimizer = optimizers.Adam(lr=learning_rate)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
        
            print('Architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n', model.summary())
        else:
            x1 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
            x1 = BatchNormalization()(x1)
            x1 = Activation('relu')(x1)
            x1 = Dropout(0.5)(x1)
            output1 = Dense(2, activation='linear', kernel_regularizer=l2(), name='SNR')(x1)
    
            x2 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
            x2 = BatchNormalization()(x2)
            x2 = Activation('relu')(x2)
            x2 = Dropout(0.5)(x2)
            output2 = Dense(1, activation='tanh', kernel_regularizer=l2(), name='S')(x2)
    
            x3 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x)
            x3 = BatchNormalization()(x3)
            x3 = Activation('relu')(x3)
            x3 = Dropout(0.5)(x3)
            output3 = Dense(1, activation='tanh', kernel_regularizer=l2(), name='M')(x3)
            
            x4 = Concatenate(axis=-1)([output1, output2, output3])
            x4 = Dense(512, kernel_regularizer=l2(), kernel_initializer=he_uniform(0))(x4)
            x4 = BatchNormalization()(x4)
            x4 = Activation('relu')(x4)
            x4 = Dropout(0.5)(x4)
            output4 = Dense(3, activation='softmax', kernel_regularizer=l2(), name='3C')(x4)
        
            model = Model(input_img, [output1, output2, output3, output4])
            learning_rate = 0.0001
            optimizer = optimizers.Nadam(lr=learning_rate)
            model.compile(
                loss={'SNR':'mean_squared_error', 'S': 'hinge', 'M': 'hinge', '3C':'categorical_crossentropy'}, 
                loss_weights={'SNR':1, 'S':1, 'M':1, '3C':1}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )
        
            print('Multi-task learning architecture (adapted from CNN model of Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION)\n', model.summary())
        
    
        
    
    elif PARAMS['CNN_architecture_name'] == 'Papakostas_et_al':
       
        '''
        Baseline :- papakostas_et_al
        '''    
        input_img = Input(PARAMS['input_shape'])    
    
        x = Conv2D(96, kernel_size=(5, 5), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(input_img)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        
        x = Flatten()(x)
        
        x = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
    

        if not PARAMS['multi_task_learning']:
            x = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
        
            output = Dense(len(PARAMS['classes'])+1, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
        
            model = Model(input_img, output)
            learning_rate = 0.001
            optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9)
            model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])    
        
            print('Architecture proposed by Papakostas et al. Expert Systems with Applications 2018\n', model.summary())
    
        else:
            x1 = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
            x1 = Activation('relu')(x1)
            x1 = Dropout(0.5)(x1)
            output1 = Dense(2, activation='linear', kernel_regularizer=l2(), name='SNR')(x1)
    
            output2 = Dense(1, activation='tanh', kernel_regularizer=l2(), name='S')(x1)
    
            output3 = Dense(1, activation='tanh', kernel_regularizer=l2(), name='M')(x1)
            
            x4 = Concatenate(axis=-1)([output1, output2, output3])
            output4 = Dense(3, activation='softmax', kernel_regularizer=l2(), name='3C')(x4)
        
            model = Model(input_img, [output1, output2, output3, output4])
            learning_rate = 0.0001
            optimizer = optimizers.Nadam(lr=learning_rate)
            model.compile(
                loss={'SNR':'mean_squared_error', 'S': 'hinge', 'M': 'hinge', '3C':'categorical_crossentropy'}, 
                loss_weights={'SNR':1, 'S':1, 'M':1, '3C':1}, 
                optimizer=optimizer, 
                metrics={'3C':'accuracy'}
                )
        
            print('Multi-task learning architecture (adapted from CNN model of Papakostas et al. Expert Systems with Applications 2018)\n', model.summary())

    
    return model, learning_rate



def test_file_wise_generator(PARAMS, file_name1, file_name2, target_dB):
    if file_name2=='':
        fv = preproc.get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], file_name1, '', target_dB)
    elif file_name1=='':
        fv = preproc.get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', file_name2, target_dB)
    else:
        fv = preproc.get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], file_name1, file_name2, target_dB)
        
    batchData = preproc.get_feature_patches(PARAMS, fv, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['featName'])
    numLab = np.shape(batchData)[0]
    
    if not PARAMS['multi_task_learning']:
        if file_name2=='':
            batchLabel = np.array([1]*numLab)
        elif file_name1=='':
            batchLabel = np.array([0]*numLab)
        else:
            batchLabel = np.array([2]*numLab)
        OHE_batchLabel = to_categorical(batchLabel, num_classes=3)
        
        return batchData, OHE_batchLabel
    else:
        '''
        Pure Impure
        '''
        batchLabel1 = np.ones((numLab,2))
        if file_name2=='': # Speech
            batchLabel1 = np.repeat(np.array([0, 1], ndmin=2), numLab, axis=0)
        elif file_name1=='': # Music
            batchLabel1 = np.repeat(np.array([1, 0], ndmin=2), numLab, axis=0)
        else: # speech-music
            if target_dB>=0:
                batchLabel1 = np.repeat(np.array([1/np.power(10,(target_dB/10)), 1], ndmin=2), numLab, axis=0)
            else:
                batchLabel1 = np.repeat(np.array([1, np.power(10,(target_dB/10))], ndmin=2), numLab, axis=0)                
    
        '''
        Speech Nonspeech
        '''
        batchLabel2 = np.ones(numLab)*-1
        if file_name2=='': # Speech
            batchLabel2 = np.ones(numLab)
    
        '''
        Music Nonmusic
        '''
        batchLabel3 = np.ones(numLab)*-1
        if file_name1=='': # Music
            batchLabel3 = np.ones(numLab)

        '''
        Multi-class
        '''
        batchLabel4 = np.ones(numLab)
        if file_name2=='': # Speech
            batchLabel4 = np.ones(numLab)*1
        elif file_name1=='': # music
            batchLabel4 = np.ones(numLab)*0
        else: # Speech_Music
            batchLabel4 = np.ones(numLab)*2
        OHE_batchLabel4 = to_categorical(batchLabel4, num_classes=3)

        batchLabel = {'SNR': batchLabel1, 'S': batchLabel2, 'M': batchLabel3, '3C': OHE_batchLabel4}

    return batchData, batchLabel



def test_model_generator(PARAMS, Train_Params):
    testingTimeTaken = 0
        
    start = time.clock()
    if not os.path.exists(PARAMS['opDir']+'/evaluate_generator_results_fold'+str(PARAMS['fold'])+'.pkl'):
        metrics = Train_Params['model'].evaluate(
                generator(PARAMS, PARAMS['folder'], PARAMS['test_files'], PARAMS['batch_size']),
                steps=PARAMS['test_steps'], 
                verbose=1,
                )
        if PARAMS['save_flag']:
            misc.save_obj(metrics, PARAMS['opDir'], 'evaluate_generator_results_fold'+str(PARAMS['fold']))
    else:
        metrics = misc.load_obj(PARAMS['opDir'], 'evaluate_generator_results_fold'+str(PARAMS['fold']))
    metric_names = Train_Params['model'].metrics_names
    
    print(metric_names)
    print(metrics)
    testingTimeTaken = time.clock() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    return metrics, metric_names, testingTimeTaken




    
def test_model(PARAMS, Train_Params, target_dB):
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])

    PtdLabels_mu = []
    GroundTruth_mu = []

    PtdLabels_sp = []
    GroundTruth_sp = []

    GroundTruth_smr = np.empty([])
    Predictions_smr = np.empty([])

    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    startTime = time.clock()
    for classname in class_labels:
        files = PARAMS['test_files'][classname]
        for fl in files:
            fName = PARAMS['folder'] + '/'+ classname + '/' + fl
            if not os.path.exists(fName):
                continue
            count += 1
            if PARAMS['multi_task_learning']:
                if classname=='speech':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, fName, '', target_dB)
                    SNR_pred, S_pred, M_pred, multiclass_pred = Train_Params['model'].predict(x=batchData)
                    GroundTruth.extend([1]*len(S_pred))

                    GroundTruth_mu.extend([0]*len(S_pred))
                    GroundTruth_sp.extend([1]*len(S_pred))
                    gt_smr = np.repeat(np.array([0, 1], ndmin=2), np.shape(batchData)[0], axis=0)
                    if np.size(GroundTruth_smr)<=1:
                        GroundTruth_smr = gt_smr
                    else:
                        GroundTruth_smr = np.append(GroundTruth_smr, gt_smr, axis=0)
                    
                elif classname=='music':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, '', fName, target_dB)
                    SNR_pred, S_pred, M_pred, multiclass_pred = Train_Params['model'].predict(x=batchData)
                    GroundTruth.extend([0]*len(M_pred))

                    GroundTruth_mu.extend([1]*len(M_pred))
                    GroundTruth_sp.extend([0]*len(M_pred))
                    gt_smr = np.repeat(np.array([1, 0], ndmin=2), np.shape(batchData)[0], axis=0)
                    if np.size(GroundTruth_smr)<=1:
                        GroundTruth_smr = gt_smr
                    else:
                        GroundTruth_smr = np.append(GroundTruth_smr, gt_smr, axis=0)
                
                if np.size(Predictions)<=1:
                    Predictions = multiclass_pred
                    Predictions_smr = SNR_pred
                else:
                    Predictions = np.append(Predictions, multiclass_pred, 0)
                    Predictions_smr = np.append(Predictions_smr, SNR_pred, axis=0)
                '''
                Decision strategy
                '''
                pred_lab = np.argmax(multiclass_pred, axis=1)
                pred_lab_mu = (M_pred>0).astype(int)
                pred_lab_sp = (S_pred>0).astype(int)
                print('\nDecision: ', np.shape(M_pred), np.shape(S_pred), np.shape(pred_lab_mu), np.shape(pred_lab_sp))
            else:
                if classname=='speech':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, fName, '', target_dB)
                    pred = Train_Params['model'].predict(x=batchData)
                    GroundTruth.extend([1]*np.shape(pred)[0])
                elif classname=='music':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, '', fName, target_dB)
                    pred = Train_Params['model'].predict(x=batchData)
                    GroundTruth.extend([0]*np.shape(pred)[0])
                pred_lab = np.argmax(pred, axis=1)
                if np.size(Predictions)<=1:
                    Predictions = pred
                else:
                    Predictions = np.append(Predictions, pred, 0)
                
            PtdLabels.extend(pred_lab.flatten())
            if PARAMS['multi_task_learning']:
                PtdLabels_mu.extend(pred_lab_mu.flatten())
                PtdLabels_sp.extend(pred_lab_sp.flatten())
                if classname=='speech':
                    clLab = 1
                elif classname=='music':
                    clLab = 0
                print('MTL Sp acc:', np.round(np.sum(np.array(pred_lab_sp)==clLab)*100/len(pred_lab_sp)), np.round(np.sum(np.array(PtdLabels_sp)==np.array(GroundTruth_sp))*100/len(PtdLabels_sp)))
                print('MTL Mu acc:', np.round(np.sum(np.array(pred_lab_mu)==(1-clLab))*100/len(pred_lab_mu)), np.round(np.sum(np.array(PtdLabels_mu)==np.array(GroundTruth_mu))*100/len(PtdLabels_mu)))
                print('MTL SMR MSE:', np.mean(np.power(Predictions_smr-GroundTruth_smr,2)))
            
            print(target_dB, 'dB\t', classname, 'pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), end='\t', flush=True)
            if classname=='speech':
                print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', np.round(np.sum(np.array(pred_lab)==1)*100/len(pred_lab), 4), np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4))
            elif classname=='music':
                print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', np.round(np.sum(np.array(pred_lab)==0)*100/len(pred_lab), 4), np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4))


    files_sp = PARAMS['test_files']['speech']
    files_mu = PARAMS['test_files']['music']
    for fl_sp in files_sp:
        fl_mu = files_mu[np.random.randint(len(files_mu))]
        fName_sp = PARAMS['folder'] + '/speech/' + fl_sp
        fName_mu = PARAMS['folder'] + '/music/' + fl_mu
        count += 1
        batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, fName_mu, target_dB)
        if PARAMS['multi_task_learning']:
            SNR_pred, S_pred, M_pred, multiclass_pred = Train_Params['model'].predict(x=batchData)
            GroundTruth.extend([2]*len(S_pred))
            Predictions = np.append(Predictions, multiclass_pred, 0) # np.append(Predictions, pure_pred, 0)
            Predictions_smr = np.append(Predictions_smr, SNR_pred, axis=0)

            GroundTruth_mu.extend([0]*len(S_pred))
            GroundTruth_sp.extend([0]*len(S_pred))
            if target_dB>=0:
                gt_smr = np.repeat(np.array([1/np.power(10,(target_dB/10)), 1], ndmin=2), np.shape(batchData)[0], axis=0)
            else:
                gt_smr = np.repeat(np.array([1, np.power(10,(target_dB/10))], ndmin=2), np.shape(batchData)[0], axis=0)
            if np.size(GroundTruth_smr)<=1:
                GroundTruth_smr = gt_smr
            else:
                GroundTruth_smr = np.append(GroundTruth_smr, gt_smr, axis=0)

            '''
            Decision strategy
            '''
            pred_lab = np.argmax(multiclass_pred, axis=1)
            pred_lab_mu = (M_pred>0).astype(int)
            pred_lab_sp = (S_pred>0).astype(int)

        else:
            pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)
            Predictions = np.append(Predictions, pred, 0)
            GroundTruth.extend([2]*np.shape(pred)[0])
            Predictions = np.append(Predictions, pred, 0)

        PtdLabels.extend(pred_lab.flatten())
        if PARAMS['multi_task_learning']:
            PtdLabels_mu.extend(pred_lab_mu.flatten())
            PtdLabels_sp.extend(pred_lab_sp.flatten())
            print('MTL Sp acc:', np.round(np.sum(np.array(pred_lab_sp)==0)*100/len(pred_lab_sp)), np.round(np.sum(np.array(PtdLabels_sp)==np.array(GroundTruth_sp))*100/len(PtdLabels_sp)))
            print('MTL Mu acc:', np.round(np.sum(np.array(pred_lab_mu)==0)*100/len(pred_lab_mu)), np.round(np.sum(np.array(PtdLabels_mu)==np.array(GroundTruth_mu))*100/len(PtdLabels_mu)))
            print('MTL SMR MSE:', np.mean(np.power(Predictions_smr-GroundTruth_smr,2)))

        print(target_dB, 'dB\tspeech_music pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), fl_sp, fl_mu, np.shape(batchData), ' acc=', np.round(np.sum(np.array(pred_lab)==2)*100/len(pred_lab), 4), np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4))

    testingTimeTaken = time.clock() - startTime
    print('Time taken for model testing: ',testingTimeTaken)
    ConfMat, fscore = misc.getPerformance(PtdLabels, GroundTruth)
    
    fscore_mu = []
    fscore_sp = []
    MSE_SMR = 0
    if PARAMS['multi_task_learning']:
        print('Music labels: ', np.shape(PtdLabels_mu), np.shape(GroundTruth_mu))
        print('Speech labels: ', np.shape(PtdLabels_sp), np.shape(GroundTruth_sp))
        
        ConfMat_mu, fscore_mu = misc.getPerformance_binary(PtdLabels_mu, GroundTruth_mu)
        ConfMat_sp, fscore_sp = misc.getPerformance_binary(PtdLabels_sp, GroundTruth_sp)
        MSE_SMR = np.mean(np.power(Predictions_smr-GroundTruth_smr,2))
    
    return ConfMat, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken, fscore_mu, fscore_sp, MSE_SMR




def test_model_ensemble(PARAMS, Train_Params_H, Train_Params_P, target_dB):
    PtdLabels_ensemble = []
    GroundTruth_ensemble = []
    Predictions_H = np.empty([])
    Predictions_P = np.empty([])
    Predictions_ensemble = np.empty([])

    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    startTime = time.clock()
    for classname in class_labels:
        files = PARAMS['test_files'][classname]
        for fl in files:
            fName = PARAMS['folder'] + '/'+ classname + '/' + fl
            if not os.path.exists(fName):
                continue
            count += 1
            if PARAMS['multi_task_learning']:
                PARAMS['HPSS_type'] = 'Harmonic'
                batchData_H, batchLabel_H = test_file_wise_generator(PARAMS, fName, '', target_dB)
                SNR_pred_H, S_pred_H, M_pred_H, multiclass_pred_H = Train_Params_H['model'].predict(x=batchData_H)

                PARAMS['HPSS_type'] = 'Percussive'
                batchData_P, batchLabel_P = test_file_wise_generator(PARAMS, fName, '', target_dB)
                SNR_pred_P, S_pred_P, M_pred_P, multiclass_pred_P = Train_Params_P['model'].predict(x=batchData_P)

                if classname=='speech':
                    GroundTruth_ensemble.extend([1]*len(S_pred_H))
                elif classname=='music':
                    GroundTruth_ensemble.extend([0]*len(M_pred_H))

                if np.size(Predictions_H)<=1:
                    Predictions_H = multiclass_pred_H
                    Predictions_P = multiclass_pred_P
                else:
                    Predictions_H = np.append(Predictions_H, multiclass_pred_H, 0)
                    Predictions_P = np.append(Predictions_P, multiclass_pred_P, 0)
                
                multiclass_pred_comb = multiclass_pred_H + multiclass_pred_P
                pred_lab = np.argmax(multiclass_pred_comb, axis=1)
            else:
                PARAMS['HPSS_type'] = 'Harmonic'
                batchData_H, batchLabel_H = test_file_wise_generator(PARAMS, fName, '', target_dB)
                pred_H = Train_Params_H['model'].predict(x=batchData_H)

                PARAMS['HPSS_type'] = 'Percussive'
                batchData_P, batchLabel_P = test_file_wise_generator(PARAMS, fName, '', target_dB)
                pred_P = Train_Params_P['model'].predict(x=batchData_P)
                
                if classname=='speech':
                    GroundTruth_ensemble.extend([1]*np.shape(pred_H)[0])
                elif classname=='music':
                    GroundTruth_ensemble.extend([0]*np.shape(pred_P)[0])

                pred_comb = pred_H + pred_P
                pred_lab = np.argmax(pred_comb, axis=1)
                if np.size(Predictions_H)<=1:
                    Predictions_H = pred_H
                    Predictions_P = pred_P
                else:
                    Predictions_H = np.append(Predictions_H, pred_H, 0)
                    Predictions_P = np.append(Predictions_P, pred_P, 0)
                
            PtdLabels_ensemble.extend(pred_lab.flatten())
            
            print(target_dB, 'dB\t', classname, 'pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), end='\t', flush=True)
            if classname=='speech':
                print(fl, np.shape(batchData_H), len(PtdLabels_ensemble), len(GroundTruth_ensemble), ' acc=', np.round(np.sum(np.array(pred_lab)==1)*100/len(pred_lab), 4), np.round(np.sum(np.array(PtdLabels_ensemble)==np.array(GroundTruth_ensemble))*100/len(PtdLabels_ensemble), 4))
            elif classname=='music':
                print(fl, np.shape(batchData_H), len(PtdLabels_ensemble), len(GroundTruth_ensemble), ' acc=', np.round(np.sum(np.array(pred_lab)==0)*100/len(pred_lab), 4), np.round(np.sum(np.array(PtdLabels_ensemble)==np.array(GroundTruth_ensemble))*100/len(PtdLabels_ensemble), 4))


    files_sp = PARAMS['test_files']['speech']
    files_mu = PARAMS['test_files']['music']
    for fl_sp in files_sp:
        fl_mu = files_mu[np.random.randint(len(files_mu))]
        fName_sp = PARAMS['folder'] + '/speech/' + fl_sp
        fName_mu = PARAMS['folder'] + '/music/' + fl_mu
        count += 1
        PARAMS['HPSS_type'] = 'Harmonic'
        batchData_H, batchLabel_H = test_file_wise_generator(PARAMS, fName_sp, fName_mu, target_dB)
        PARAMS['HPSS_type'] = 'Percussive'
        batchData_P, batchLabel_P = test_file_wise_generator(PARAMS, fName_sp, fName_mu, target_dB)
        if PARAMS['multi_task_learning']:
            SNR_pred_H, S_pred_H, M_pred_H, multiclass_pred_H = Train_Params_H['model'].predict(x=batchData_H)
            SNR_pred_P, S_pred_P, M_pred_P, multiclass_pred_P = Train_Params_P['model'].predict(x=batchData_P)
            GroundTruth_ensemble.extend([2]*len(S_pred_H))
            Predictions_H = np.append(Predictions_H, multiclass_pred_H, 0)
            Predictions_P = np.append(Predictions_P, multiclass_pred_P, 0)
            multiclass_pred_comb = multiclass_pred_H + multiclass_pred_P
            pred_lab = np.argmax(multiclass_pred_comb, axis=1)
        else:
            pred_H = Train_Params_H['model'].predict(x=batchData_H)
            pred_P = Train_Params_P['model'].predict(x=batchData_P)
            pred_comb = pred_H + pred_P
            pred_lab = np.argmax(pred_comb, axis=1)
            Predictions_H = np.append(Predictions_H, pred_H, 0)
            Predictions_P = np.append(Predictions_P, pred_P, 0)
            GroundTruth_ensemble.extend([2]*np.shape(pred_H)[0])

        PtdLabels_ensemble.extend(pred_lab.flatten())

        print(target_dB, 'dB\tspeech_music pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), fl_sp, fl_mu, np.shape(batchData_H), ' acc=', np.round(np.sum(np.array(pred_lab)==2)*100/len(pred_lab), 4), np.round(np.sum(np.array(PtdLabels_ensemble)==np.array(GroundTruth_ensemble))*100/len(PtdLabels_ensemble), 4))

    Predictions_ensemble = Predictions_H + Predictions_P
    testingTimeTaken = time.clock() - startTime
    print('Time taken for model testing: ',testingTimeTaken)
    ConfMat_ensemble, fscore_ensemble = misc.getPerformance(PtdLabels_ensemble, GroundTruth_ensemble)
    
    return ConfMat_ensemble, fscore_ensemble, PtdLabels_ensemble, Predictions_ensemble, GroundTruth_ensemble, testingTimeTaken
