#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:24:28 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import sys
import datetime
import lib.misc as misc
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
import time
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from lib.preprocessing import load_and_preprocess_signal, mix_signals
import librosa
from sklearn.preprocessing import StandardScaler
from lib.cython_impl.tools import extract_patches as cextract_patches
from tcn import TCN
from tcn.tcn import process_dilations
from tensorflow.keras.layers import BatchNormalization, Dropout, Input, Activation, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2





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




def get_Lemaire_model(
        TR_STEPS, 
        kernel_size=3, # Temporal Conv, LogMelSpec
        Nd=8, # Temporal Conv, LogMelSpec
        nb_stacks=3, # Temporal Conv, LogMelSpec
        n_layers=1, # Temporal Conv, LogMelSpec
        n_filters=32, # Temporal Conv, LogMelSpec
        use_skip_connections=False, # Temporal Conv, LogMelSpec
        activation='norm_relu', 
        bidirectional=True, 
        N_MELS=120, 
        n_classes=5, 
        patch_size=68,
        ):
    '''
    TCN based model architecture proposed by Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection    

    Parameters
    ----------
    TR_STEPS : int
        Number of training batches per epoch.
    n_filters : int, optional
        The default is 32.
    Nd : int, optional
        The default is 3.
    kernel_size : int, optional
        The default is 3.
    nb_stacks : int, optional
        The default is 10.
    activation : string, optional
        The default is 'norm_relu'.
    n_layers : int, optional
        The default is 3.
    use_skip_connections : boolean, optional
        The default is False.
    bidirectional : boolean, optional
        The default is True.
    N_MELS : int, optional
        The default is 120.
    n_classes : int, optional
        The default is 5.
    patch_size : int, optional
        The default is 68.

    Returns
    -------
    model : tensorflow.keras.models.Model
        CNN model.
    lr : float
        Learning rate.

    '''
    dilations = [2**nd for nd in range(Nd)]
    list_n_filters = [n_filters]*n_layers
    dropout_rate = np.random.uniform(0.05,0.5)
    bidirectional = True
    
    if bidirectional:
        padding = 'same'
    else:
        padding = 'causal'

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(patch_size, N_MELS))
        
    for i in range(n_layers):
        if i == 0:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, 'norm_relu', padding, use_skip_connections, dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, 'norm_relu', padding, use_skip_connections, dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Flatten()(x)

    x = Dense(n_classes)(x)
    x = Activation('softmax')(x)
    output_layer = x
    
    model = Model(input_layer, output_layer)

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*TR_STEPS, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer=optimizer)

    print(model.summary())
    print('Architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')

    return model, initial_learning_rate





def MTL_modifications(x):
    '''
    MTL modifications to be applied to the baseline models. The modified
    architecture is tuned on the MUSAN dataset.
    
    Parameters
    ----------
    x : Keras tensor
        Prefinal layer of an architecture to which MTL is to be applied.

    Returns
    -------
    sp_output : Keras tensor
        Output layer of the speech/non-speech task.
    mu_output : Keras tensor
        Output layer of the music/non-music task.
    no_output : Keras tensor
        Output layer of the noise/non-noise task.
    smr_output : Keras tensor
        Output layer of the SMR regression task.

    '''
    #''' Speech/Non-Speech output '''
    x_sp = Dense(16, kernel_regularizer=l2())(x)
    x_sp = BatchNormalization(axis=-1)(x_sp)
    x_sp = Activation('relu')(x_sp)
    x_sp = Dropout(0.4)(x_sp)

    sp_output = Dense(1, activation='sigmoid', name='S')(x_sp)


    #''' Music/Non-Music output '''
    x_mu = Dense(16, kernel_regularizer=l2())(x)
    x_mu = BatchNormalization(axis=-1)(x_mu)
    x_mu = Activation('relu')(x_mu)
    x_mu = Dropout(0.4)(x_mu)

    x_mu = Dense(16, kernel_regularizer=l2())(x)
    x_mu = BatchNormalization(axis=-1)(x_mu)
    x_mu = Activation('relu')(x_mu)
    x_mu = Dropout(0.4)(x_mu)
    
    mu_output = Dense(1, activation='sigmoid', name='M')(x_mu)

    #''' Noise/Non-Noise output '''
    x_no = Dense(16, kernel_regularizer=l2())(x)
    x_no = BatchNormalization(axis=-1)(x_no)
    x_no = Activation('relu')(x_no)
    x_no = Dropout(0.4)(x_no)

    no_output = Dense(1, activation='sigmoid', name='N')(x_no)
    
    #''' Speech-to-Music-to-Noise Ratio (SMNR) output '''
    x_smr = Dense(16, kernel_regularizer=l2())(x)
    x_smr = BatchNormalization(axis=-1)(x_smr)
    x_smr = Activation('relu')(x_smr)
    x_smr = Dropout(0.4)(x_smr)

    x_smr = Dense(16, kernel_regularizer=l2())(x)
    x_smr = BatchNormalization(axis=-1)(x_smr)
    x_smr = Activation('relu')(x_smr)
    x_smr = Dropout(0.4)(x_smr)
    
    smr_output = Dense(3, activation='linear', name='R')(x_smr)
    
    return sp_output, x_sp, mu_output, x_mu, no_output, x_no, smr_output, x_smr




def get_Lemaire_MTL_model(
        TR_STEPS,
        N_MELS=120,
        n_classes=5,
        patch_size=68,
        loss_weights=None,
        ):
    '''
    MTL modification of the TCN based model architecture proposed by 
    Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection
    The model parameters are tuned on the MUSAN dataset.

        [3] Lemaire, Q., & Holzapfel, A. (2019). Temporal convolutional networks 
    for speech and music detection in radio broadcast. In 20th International 
    Society for Music Information Retrieval Conference, ISMIR 2019, 4-8 
    November 2019. International Society for Music Information Retrieval.

    Parameters
    ----------
    TR_STEPS : int
        Number of training batches per epoch.
    N_MELS : int, optional
        The default is 120.
    n_classes : int, optional
        The default is 5.
    patch_size : int, optional
        The default is 68.
    loss_weights : dict, optional
        The default is {'S': 1.0, 'M': 1.0, 'R': 1.0, '3C': 1.0}.

    Returns
    -------
    model : tensorflow.keras.models.Model
        CNN model.
    lr : float
        Learning rate.

    '''
    kernel_size = 3 # Temporal Conv, LogMelSpec
    Nd = 8 # Temporal Conv, LogMelSpec
    nb_stacks = 3 # Temporal Conv, LogMelSpec
    n_layers = 1 # Temporal Conv, LogMelSpec
    n_filters = 32 # Temporal Conv, LogMelSpec
    use_skip_connections = False # Temporal Conv, LogMelSpec
    activation = 'norm_relu' 
    dilations = [2**nd for nd in range(Nd)]
    list_n_filters = [n_filters]*n_layers
    dropout_rate = np.random.uniform(0.05,0.5)
    padding = 'same'
    dilations = process_dilations(dilations)
    
    input_layer = Input(shape=(patch_size,N_MELS))
        
    for i in range(n_layers):
        if i == 0:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Flatten()(x)

    classification_output = Dense(n_classes, activation='softmax', name='3C')(x)

    sp_output, x_sp, mu_output, x_mu, no_output, x_no, smr_output, x_smr = MTL_modifications(x)
    
    model = Model(input_layer, [sp_output, mu_output, no_output, smr_output, classification_output])
    # model = Model(input_layer, [sp_output, mu_output, smr_output, classification_output])

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*TR_STEPS, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    model.compile(
        loss={
            'S': 'binary_crossentropy', 
            'M': 'binary_crossentropy', 
            'N': 'binary_crossentropy', 
            'R':'mean_squared_error', 
            '3C': 'categorical_crossentropy'
            },
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )

    print(model.summary())
    print('MTL modification of the architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')

    return model, initial_learning_rate





def get_featuregram(PARAMS, classname, feature_opDir, fName_path_sp, fName_path_mu, fName_path_no, target_dB, n_fft, n_mels, featName, save_feat=True):
    if (fName_path_sp!='') and (fName_path_mu!=''): # speech_music
        fName = fName_path_sp.split('/')[-1].split('.')[0]+'_'+fName_path_mu.split('/')[-1].split('.')[0]+'_'+str(target_dB)+'dB'
    if (fName_path_sp!='') and (fName_path_no!=''): # speech_noise
        fName = fName_path_sp.split('/')[-1].split('.')[0]+'_'+fName_path_no.split('/')[-1].split('.')[0]+'_'+str(target_dB)+'dB'
    elif fName_path_sp!='': # speech
        fName = fName_path_sp.split('/')[-1].split('.')[0]
    elif fName_path_mu!='': # music
        fName = fName_path_mu.split('/')[-1].split('.')[0]
    elif fName_path_no!='': # noise
        fName = fName_path_no.split('/')[-1].split('.')[0]

    if not os.path.exists(feature_opDir+'/'+classname+'/'+fName+'.npy'):
        if classname=='speech_music':
            # print('fName_path_sp: ', fName_path_sp)
            # print('fName_path_mu: ', fName_path_mu)
            Xin_sp, fs = load_and_preprocess_signal(fName_path_sp, PARAMS['Tw'], PARAMS['Ts'])
            Xin_mu, fs = load_and_preprocess_signal(fName_path_mu, PARAMS['Tw'], PARAMS['Ts'])
            Xin = mix_signals(Xin_sp, Xin_mu, target_dB)

        if classname=='speech_noise':
            # print('fName_path_sp: ', fName_path_sp)
            # print('fName_path_no: ', fName_path_no)
            Xin_sp, fs = load_and_preprocess_signal(fName_path_sp, PARAMS['Tw'], PARAMS['Ts'])
            Xin_no, fs = load_and_preprocess_signal(fName_path_no, PARAMS['Tw'], PARAMS['Ts'])
            Xin = mix_signals(Xin_sp, Xin_no, target_dB)
    
        elif classname=='speech':
            Xin, fs = load_and_preprocess_signal(fName_path_sp, PARAMS['Tw'], PARAMS['Ts'])
    
        elif classname=='music':
            Xin, fs = load_and_preprocess_signal(fName_path_mu, PARAMS['Tw'], PARAMS['Ts'])

        elif classname=='noise':
            Xin, fs = load_and_preprocess_signal(fName_path_no, PARAMS['Tw'], PARAMS['Ts'])

        if featName=='LogMelSpec':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            fv = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False, n_mels=n_mels)
            fv = librosa.core.power_to_db(fv**2)
            fv = fv.astype(np.float32)

        elif featName.startswith('LogMelHarm') or featName.startswith('LogMelPerc'):
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            Spec = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            H_Spec, P_Spec = librosa.decompose.hpss(S=Spec, kernel_size=(PARAMS['l_harm'][PARAMS['Model']], PARAMS['l_perc'][PARAMS['Model']]))
            fv_H = librosa.feature.melspectrogram(S=H_Spec, n_mels=n_mels)
            fv_H = librosa.core.power_to_db(fv_H**2)
            fv_P = librosa.feature.melspectrogram(S=P_Spec, n_mels=n_mels)
            fv_P = librosa.core.power_to_db(fv_P**2)
            fv = np.append(fv_H, fv_P, axis=0) 
            fv = fv.astype(np.float32)

        if save_feat:
            if not os.path.exists(feature_opDir+'/'+classname+'/'):
                os.makedirs(feature_opDir+'/'+classname+'/')
            np.save(feature_opDir+'/'+classname+'/'+fName+'.npy', fv)
    else:
        try:
            fv = np.load(feature_opDir+'/'+classname+'/'+fName+'.npy', allow_pickle=True)
        except:
            print('Error loading: ', feature_opDir+'/'+classname+'/'+fName+'.npy')
            # This condition is kept so that the program terminates and erring file can be identified 
            fv = np.load(feature_opDir+'/'+classname+'/'+fName+'.npy', allow_pickle=True)
    
    return fv





def get_feature_patches(PARAMS, FV, patch_size, patch_shift, featName):
    # FV should be of the shape (nFeatures, nFrames)
    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)        

    if featName=='LogMelSpec':
        FV = FV.T
        FV = StandardScaler(copy=False).fit_transform(FV)
        FV = FV.T
        patches = cextract_patches(FV, np.shape(FV), patch_size, patch_shift)
        if not 'Lemaire_et_al' in PARAMS['Model']:
            patches = np.expand_dims(patches, axis=3)
        
    elif featName.startswith('LogMelHarm') or featName.startswith('LogMelPerc'):
        if (featName=='LogMelHarmSpec') or (featName=='LogMelHarmPercSpec'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            FV_H = FV_H.T
            FV_H = StandardScaler(copy=False).fit_transform(FV_H)
            FV_H = FV_H.T
            patches_H = cextract_patches(FV_H, np.shape(FV_H), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_H = np.expand_dims(patches_H, axis=3)

        if (featName=='LogMelPercSpec') or (featName=='LogMelHarmPercSpec'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            FV_P = FV_P.T
            FV_P = StandardScaler(copy=False).fit_transform(FV_P)
            FV_P = FV_P.T
            patches_P = cextract_patches(FV_P, np.shape(FV_P), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_P = np.expand_dims(patches_P, axis=3)
        
        if 'HarmPerc' in featName:
            patches = np.append(patches_H, patches_P, axis=1)
        elif 'Harm' in featName:
            patches = patches_H.copy()
        elif 'Perc' in featName:
            patches = patches_P.copy()

    return patches





def generator(PARAMS, folder, file_list, batchSize):
    batch_count = 0
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])
    np.random.shuffle(file_list['noise'])

    file_list_sp_temp = file_list['speech'].copy()
    file_list_mu_temp = file_list['music'].copy()
    file_list_no_temp = file_list['noise'].copy()

    batchData_sp = np.empty([], dtype=float)
    batchData_mu = np.empty([], dtype=float)
    batchData_no = np.empty([], dtype=float)

    balance_sp = 0
    balance_mu = 0
    balance_no = 0

    if not os.path.exists(PARAMS['feature_opDir']+'/speech/'):
        os.makedirs(PARAMS['feature_opDir']+'/speech/')
    if not os.path.exists(PARAMS['feature_opDir']+'/music/'):
        os.makedirs(PARAMS['feature_opDir']+'/music/')
    if not os.path.exists(PARAMS['feature_opDir']+'/noise/'):
        os.makedirs(PARAMS['feature_opDir']+'/noise/')

    np.random.shuffle(file_list['speech+music'])
    file_list_spmu_temp = file_list['speech+music'].copy()
    batchData_spmu = np.empty([], dtype=float)
    batchData_spmu_target_dB = np.empty([], dtype=float)
    balance_spmu = 0
    if not os.path.exists(PARAMS['feature_opDir']+'/speech_music/'):
        os.makedirs(PARAMS['feature_opDir']+'/speech_music/')

    np.random.shuffle(file_list['speech+noise'])
    file_list_spno_temp = file_list['speech+noise'].copy()
    batchData_spno = np.empty([], dtype=float)
    batchData_spno_target_dB = np.empty([], dtype=float)
    balance_spno = 0
    if not os.path.exists(PARAMS['feature_opDir']+'/speech_noise/'):
        os.makedirs(PARAMS['feature_opDir']+'/speech_noise/')
        
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
            fv_sp = get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', '', None, n_fft, n_mels, featName)
            fv_sp_patches = get_feature_patches(PARAMS, fv_sp, PARAMS['W'], PARAMS['W_shift'], featName)
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
            fv_mu = get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, '', None, n_fft, n_mels, featName)
            fv_mu_patches = get_feature_patches(PARAMS, fv_mu, PARAMS['W'], PARAMS['W_shift'], featName)
            if balance_mu==0:
                batchData_mu = fv_mu_patches
            else:
                batchData_mu = np.append(batchData_mu, fv_mu_patches, axis=0)
            balance_mu += np.shape(fv_mu_patches)[0]
            # print('Music: ', batchSize, balance_mu, np.shape(batchData_mu))


        while balance_no<batchSize:
            if not file_list_no_temp:
                file_list_no_temp = file_list['noise'].copy()
            no_fName = file_list_no_temp.pop()
            no_fName_path = folder + '/noise/' + no_fName
            if not os.path.exists(no_fName_path):
                # print(mu_fName_path, os.path.exists(mu_fName_path))
                continue
            fv_no = get_featuregram(PARAMS, 'noise', PARAMS['feature_opDir'], '', '', no_fName_path, None, n_fft, n_mels, featName)
            fv_no_patches = get_feature_patches(PARAMS, fv_no, PARAMS['W'], PARAMS['W_shift'], featName)
            if balance_no==0:
                batchData_no = fv_no_patches
            else:
                batchData_no = np.append(batchData_no, fv_no_patches, axis=0)
            balance_no += np.shape(fv_no_patches)[0]
            # print('Noie: ', batchSize, balance_no, np.shape(batchData_no))


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
            fv_spmu = get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], sp_fName_path, mu_fName_path, '', target_dB, n_fft, n_mels, featName)
            fv_spmu_patches = get_feature_patches(PARAMS, fv_spmu, PARAMS['W'], PARAMS['W_shift'], featName)
            if balance_spmu==0:
                batchData_spmu = fv_spmu_patches
                batchData_spmu_target_dB = np.array([target_dB]*np.shape(fv_spmu_patches)[0])
            else:
                batchData_spmu = np.append(batchData_spmu, fv_spmu_patches, axis=0)
                batchData_spmu_target_dB = np.append(batchData_spmu_target_dB, np.array([target_dB]*np.shape(fv_spmu_patches)[0]))
            balance_spmu += np.shape(fv_spmu_patches)[0]
            # print('SpeechMusic: ', batchSize, balance_spmu, np.shape(batchData_spmu))
            

        while balance_spno<batchSize:
            if not file_list_spno_temp:
                file_list_spno_temp = file_list['speech+noise'].copy()
            np.random.shuffle(file_list_spno_temp)
            spno_info = file_list_spno_temp.pop()
            sp_fName = spno_info['speech']
            sp_fName_path = folder + '/speech/' + sp_fName
            no_fName = spno_info['noise']
            no_fName_path = folder + '/noise/' + no_fName
            target_dB = spno_info['SMR']
            if (not os.path.exists(no_fName_path)) or (not os.path.exists(sp_fName_path)):
                continue
            fv_spno = get_featuregram(PARAMS, 'speech_noise', PARAMS['feature_opDir'], sp_fName_path, '', no_fName_path, target_dB, n_fft, n_mels, featName)
            fv_spno_patches = get_feature_patches(PARAMS, fv_spno, PARAMS['W'], PARAMS['W_shift'], featName)
            if balance_spno==0:
                batchData_spno = fv_spno_patches
                batchData_spno_target_dB = np.array([target_dB]*np.shape(fv_spno_patches)[0])
            else:
                batchData_spno = np.append(batchData_spno, fv_spno_patches, axis=0)
                batchData_spno_target_dB = np.append(batchData_spno_target_dB, np.array([target_dB]*np.shape(fv_spno_patches)[0]))
            balance_spno += np.shape(fv_spno_patches)[0]
            # print('SpeechNoise: ', batchSize, balance_spno, np.shape(batchData_spno))


        batchData = batchData_mu[:batchSize, :] # music label=0
        batchData = np.append(batchData, batchData_sp[:batchSize, :], axis=0) # speech label=1
        batchData = np.append(batchData, batchData_spmu[:batchSize, :], axis=0) # speech+music label=2
        batchData = np.append(batchData, batchData_no[:batchSize, :], axis=0) # noise label=3
        batchData = np.append(batchData, batchData_spno[:batchSize, :], axis=0) # speech+noise label=4

        balance_mu -= batchSize
        balance_sp -= batchSize
        balance_spmu -= batchSize
        balance_no -= batchSize
        balance_spno -= batchSize

        batchData_mu = batchData_mu[batchSize:, :]            
        batchData_sp = batchData_sp[batchSize:, :]
        batchData_spmu = batchData_spmu[batchSize:, :]            
        batchData_no = batchData_no[batchSize:, :]            
        batchData_spno = batchData_spno[batchSize:, :]            

        batchLabel = [0]*batchSize # music
        batchLabel.extend([1]*batchSize) # speech
        batchLabel.extend([2]*batchSize) # speech+music
        batchLabel.extend([3]*batchSize) # noise
        batchLabel.extend([4]*batchSize) # speech+noise

        batchLabel_smr = np.ones((len(PARAMS['classes'])*batchSize,3))
        batchLabel_smr[:batchSize, :] = np.repeat(np.array([1, 0, 0], ndmin=2), batchSize, axis=0) # music
        batchLabel_smr[batchSize:2*batchSize] = np.repeat(np.array([0, 1, 0], ndmin=2), batchSize, axis=0) # speech
        label_idx = 2*batchSize # speech+music
        for i in range(batchSize):
            if batchData_spmu_target_dB[i]>=0:
                batchLabel_smr[label_idx] = np.array([1/np.power(10,(batchData_spmu_target_dB[i]/10)), 1,0], ndmin=2)
            else:
                batchLabel_smr[label_idx] = np.array([1, np.power(10,(batchData_spmu_target_dB[i]/10)), 0], ndmin=2)
            label_idx += 1
        batchData_spmu_target_dB = batchData_spmu_target_dB[batchSize:]
        batchLabel_smr[3*batchSize:4*batchSize] = np.repeat(np.array([0, 0, 1], ndmin=2), batchSize, axis=0) # noise
        label_idx = 4*batchSize # speech+noise
        for i in range(batchSize):
            if batchData_spno_target_dB[i]>=0:
                batchLabel_smr[label_idx] = np.array([0,1/np.power(10,(batchData_spno_target_dB[i]/10)),1], ndmin=2)
            else:
                batchLabel_smr[label_idx] = np.array([0, 1, np.power(10,(batchData_spno_target_dB[i]/10))], ndmin=2)
            label_idx += 1
        batchData_spno_target_dB = batchData_spno_target_dB[batchSize:]
        

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
        batchLabel_sp_nsp[:batchSize] = 0 # music
        batchLabel_sp_nsp[batchSize:2*batchSize] = 1 # speech
        batchLabel_sp_nsp[2*batchSize:3*batchSize] = 1 # speech+music
        batchLabel_sp_nsp[3*batchSize:4*batchSize] = 0 # noise
        batchLabel_sp_nsp[4*batchSize:] = 1 # speech+noise

        '''
        Music Nonmusic
        '''
        batchLabel_mu_nmu = np.copy(batchLabel)
        batchLabel_mu_nmu[:batchSize] = 1 # music
        batchLabel_mu_nmu[batchSize:2*batchSize] = 0 # speech
        batchLabel_mu_nmu[2*batchSize:] = 1 # speech+music
        batchLabel_mu_nmu[3*batchSize:4*batchSize] = 0 # noise
        batchLabel_mu_nmu[4*batchSize:] = 0 # speech+noise

        '''
        Noise Nonnoise
        '''
        batchLabel_no_nno = np.copy(batchLabel)
        batchLabel_no_nno[:batchSize] = 0 # music
        batchLabel_no_nno[batchSize:2*batchSize] = 0 # speech
        batchLabel_no_nno[2*batchSize:] = 0 # speech+music
        batchLabel_no_nno[3*batchSize:4*batchSize] = 1 # noise
        batchLabel_no_nno[4*batchSize:] = 1 # speech+noise

        batchLabel_MTL = {'R': batchLabel_smr, 'S': batchLabel_sp_nsp, 'M': batchLabel_mu_nmu, 'N':batchLabel_no_nno, '3C': OHE_batchLabel}
        # batchLabel_MTL = {'R': batchLabel_smr, 'S': batchLabel_sp_nsp, 'M': batchLabel_mu_nmu, '3C': OHE_batchLabel}
    
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(OHE_batchLabel))
        
        if 'MTL' in PARAMS['Model']:
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
                N_MELS=PARAMS['input_shape'][PARAMS['Model']][1]
                )
            
        elif PARAMS['Model']=='Lemaire_et_al_MTL':
            model, learning_rate = get_Lemaire_MTL_model(
                TR_STEPS=PARAMS['TR_STEPS'],
                N_MELS=PARAMS['input_shape'][PARAMS['Model']][1]
                )

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
                loss={'R':'mean_squared_error', 'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'N': 'binary_crossentropy', '3C': 'categorical_crossentropy'}, 
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





def test_file_wise_generator(PARAMS, file_name_sp, file_name_mu, file_name_no, target_dB):
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]

    if (file_name_sp!='') and (file_name_mu!=''): # speech_music
        fv = get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], file_name_sp, file_name_mu, '', target_dB, n_fft, n_mels, featName, save_feat=False)

    elif (file_name_sp!='') and (file_name_no!=''): # speech_noise
        fv = get_featuregram(PARAMS, 'speech_noise', PARAMS['feature_opDir'], file_name_sp, '', file_name_no, target_dB, n_fft, n_mels, featName, save_feat=False)

    elif file_name_sp!='': # speech
        fv = get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], file_name_sp, '', '', None, n_fft, n_mels, featName, save_feat=False)

    elif file_name_mu!='': # music
        fv = get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', file_name_mu, '', None, n_fft, n_mels, featName, save_feat=False)

    elif file_name_no!='': # noise
        fv = get_featuregram(PARAMS, 'noise', PARAMS['feature_opDir'], '', '', file_name_no, None, n_fft, n_mels, featName, save_feat=False)

    batchData = get_feature_patches(PARAMS, fv, PARAMS['W'], PARAMS['W_shift'], featName)

    if 'Lemaire_et_al' in PARAMS['Model']:
        batchData = np.transpose(batchData, axes=(0,2,1)) # TCN input shape=(batch_size, timesteps, ndim)

    numLab = np.shape(batchData)[0]

    if (file_name_sp!='') and (file_name_mu!=''): # speech_music
        batchLabel = np.array([2]*numLab)
    if (file_name_sp!='') and (file_name_no!=''): # speech_noise
        batchLabel = np.array([4]*numLab)
    elif file_name_sp!='': # speech
        batchLabel = np.array([1]*numLab)
    elif file_name_mu!='': # music
        batchLabel = np.array([0]*numLab)
    elif file_name_no!='': # noise
        batchLabel = np.array([3]*numLab)
    
    OHE_batchLabel = to_categorical(batchLabel, num_classes=len(PARAMS['classes']))
    
    return batchData, OHE_batchLabel



def test_model(PARAMS, Train_Params, target_dB):
    PtdLabels = np.empty([])
    GroundTruth = np.empty([])
    Predictions = np.empty([])

    startTime = time.process_time()
    if target_dB==None:
        # class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        for classname in ['music', 'speech', 'noise']:
            files = PARAMS['test_files'][classname]
            fl_count = 0 
            for fl in files:
                fl_count += 1
                fName = PARAMS['folder'] + '/'+ classname + '/' + fl
                if not os.path.exists(fName):
                    continue
                if classname=='speech':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, fName, '', '', None)
                elif classname=='music':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, '', fName, '', None)
                elif classname=='noise':
                    batchData, batchLabel = test_file_wise_generator(PARAMS, '', '', fName, None)
                
                if 'MTL' in PARAMS['Model']:
                    sp_pred, mu_pred, no_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
                    # sp_pred, mu_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
                    pred_lab = np.argmax(pred, axis=1)
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
                    elif classname=='noise':
                        GroundTruth = np.array([3]*np.shape(pred)[0])
                else:
                    Predictions = np.append(Predictions, pred, 0)
                    PtdLabels = np.append(PtdLabels, pred_lab)
                    if classname=='speech':
                        GroundTruth = np.append(GroundTruth, np.array([1]*np.shape(pred)[0]))
                    elif classname=='music':
                        GroundTruth = np.append(GroundTruth, np.array([0]*np.shape(pred)[0]))
                    elif classname=='noise':
                        GroundTruth = np.append(GroundTruth, np.array([3]*np.shape(pred)[0]))
                
                print(fl_count, '/', len(files), target_dB, 'dB\t', classname, 'pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), np.sum(pred_lab==3), np.sum(pred_lab==4), end='\t', flush=True)
                if classname=='speech':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==1)*100/len(pred_lab), 4)
                    acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
                    print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', acc_fl, acc_all)
                elif classname=='music':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==0)*100/len(pred_lab), 4)
                    acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
                    print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', acc_fl, acc_all)
                elif classname=='noise':
                    acc_fl = np.round(np.sum(np.array(pred_lab)==3)*100/len(pred_lab), 4)
                    acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
                    print(fl, np.shape(batchData), len(PtdLabels), len(GroundTruth), ' acc=', acc_fl, acc_all)

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
            batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, fName_mu, '', spmu_info['SMR'])
        else:
            batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, fName_mu, '', target_dB)

        if 'MTL' in PARAMS['Model']:
            sp_pred, mu_pred, no_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
            # sp_pred, mu_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)
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
        acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
        if target_dB==None:
            print(fl_count, '/', len(files_spmu), spmu_info['SMR'], 'dB\tspeech_music pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), np.sum(pred_lab==3), np.sum(pred_lab==4), fl_sp, fl_mu, np.shape(batchData), ' acc=', acc_fl, acc_all)
        else:
            print(fl_count, '/', len(files_spmu), target_dB, 'dB\tspeech_music pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), np.sum(pred_lab==3), np.sum(pred_lab==4), fl_sp, fl_mu, np.shape(batchData), ' acc=', acc_fl, acc_all)


    files_spno = PARAMS['test_files']['speech+noise']
    fl_count = 0
    for spno_info in files_spno:
        fl_count += 1
        fl_sp = spno_info['speech']
        fl_no = spno_info['noise']
        fName_sp = PARAMS['folder'] + '/speech/' + fl_sp
        fName_no = PARAMS['folder'] + '/noise/' + fl_no
        if target_dB==None:
            # Annotated SMR is not used in the testing function if target_dB 
            # is None so that the performance can be tested at specific 
            # SMR values
            batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, '', fName_no, spno_info['SMR'])
        else:
            batchData, batchLabel = test_file_wise_generator(PARAMS, fName_sp, '', fName_no, target_dB)

        if 'MTL' in PARAMS['Model']:
            sp_pred, mu_pred, no_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
            # sp_pred, mu_pred, smr_pred, pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)
        else:
            pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)

        if np.size(Predictions)<=1:
            Predictions = pred
            GroundTruth = np.array([4]*np.shape(pred)[0])
            PtdLabels = pred_lab
        else:
            Predictions = np.append(Predictions, pred, 0)
            GroundTruth = np.append(GroundTruth, np.array([4]*np.shape(pred)[0]))
            PtdLabels = np.append(PtdLabels, pred_lab)
        acc_fl = np.round(np.sum(np.array(pred_lab)==4)*100/len(pred_lab), 4)
        acc_all = np.round(np.sum(np.array(PtdLabels)==np.array(GroundTruth))*100/len(PtdLabels), 4)
        if target_dB==None:
            print(fl_count, '/', len(files_spno), spno_info['SMR'], 'dB\tspeech_noise pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), np.sum(pred_lab==3), np.sum(pred_lab==4), fl_sp, fl_no, np.shape(batchData), ' acc=', acc_fl, acc_all)
        else:
            print(fl_count, '/', len(files_spno), target_dB, 'dB\tspeech_noise pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1), np.sum(pred_lab==2), np.sum(pred_lab==3), np.sum(pred_lab==4), fl_sp, fl_no, np.shape(batchData), ' acc=', acc_fl, acc_all)


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
    classes = {'music':0, 'speech':1, 'speech+music':2, 'noise':3, 'speech+noise':4}
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
    patch_size = 68
    patch_shift = 68
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
            'classes': {0:'music', 1:'speech', 2:'speech_music', 3:'noise', 4:'speech_noise'},
            'data_augmentation_with_noise': True,
            'Model': 'Lemaire_et_al_MTL', # Lemaire_et_al, Lemaire_et_al_MTL
            'featName': {
                'Lemaire_et_al':'LogMelSpec',
                'Lemaire_et_al_MTL':'LogMelHarmPercSpec',
                } ,
            'n_fft': {
                'Lemaire_et_al':400, 
                'Lemaire_et_al_MTL':400, 
                },
            'n_mels': {
                'Lemaire_et_al':opt_n_mels, 
                'Lemaire_et_al_MTL':opt_n_mels, 
                },
            'l_harm': {
                'Lemaire_et_al':opt_l_harm, 
                'Lemaire_et_al_MTL':opt_l_harm, 
                },
            'l_perc': {
                'Lemaire_et_al':opt_l_perc,
                'Lemaire_et_al_MTL':opt_l_perc,
                },
            'input_shape': {
                'Lemaire_et_al':(patch_size,opt_n_mels,1), 
                'Lemaire_et_al_MTL':(patch_size,opt_n_mels,1), 
                },
            'W':patch_size,
            'W_shift':patch_shift,
            'Tw': 25,
            'Ts': 10,
            'test_SMR_levels': [-5,0,5,10,15,20],
            'dB_wise_test': False,
            }
    
    PARAMS['dataset_name_train'] = list(filter(None,PARAMS['folder'].split('/')))[-1]
    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name_train'] + '_5_class/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list_5_class.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list_5_class')
    
    n_classes = len(PARAMS['classes'])
    DT_SZ = 0
    for clNum in PARAMS['classes'].keys():
        classname = PARAMS['classes'][clNum]
        if classname=='speech_music':
            classname = 'speech+music'
        if classname=='speech_noise':
            classname = 'speech+noise'
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
            
    PARAMS['opDir'] = './results/' + PARAMS['dataset_name_train'] + '/Rebuttal_Work/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '_' + str(len(PARAMS['classes'])) + 'classes_' + PARAMS['today'] + '/' 

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
            if 'MTL' in PARAMS['Model']:
                metrics_names = ['loss', 'S_loss', 'M_loss', 'N_loss', 'R_loss', '3C_loss', '3C_accuracy']
                # metrics_names = ['loss', 'S_loss', 'M_loss', 'R_loss', '3C_loss', '3C_accuracy']
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
        if 'MTL' in PARAMS['Model']:
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
        res_dict['9'] = 'Prec_spmu:' + str(Test_Params['precision_annot'][2])
        res_dict['10'] = 'Rec_spmu:' + str(Test_Params['recall_annot'][2])
        res_dict['11'] = 'F1_spmu:' + str(Test_Params['fscore_annot'][2])
        res_dict['12'] = 'Prec_no:' + str(Test_Params['precision_annot'][3])
        res_dict['13'] = 'Rec_no:' + str(Test_Params['recall_annot'][3])
        res_dict['14'] = 'F1_no:' + str(Test_Params['fscore_annot'][3])
        res_dict['15'] = 'Prec_spno:' + str(Test_Params['precision_annot'][4])
        res_dict['16'] = 'Rec_spno:' + str(Test_Params['recall_annot'][4])
        res_dict['17'] = 'F1_spno:' + str(Test_Params['fscore_annot'][4])
        misc.print_results(PARAMS, '', res_dict)
        
        if PARAMS['dB_wise_test']:
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
            res_dict['9'] = 'Prec_spmu:'+str(Test_Params['precision_All'][2])
            res_dict['10'] = 'Rec_spmu:'+str(Test_Params['recall_All'][2])
            res_dict['11'] = 'F1_spmu:'+str(Test_Params['fscore_All'][2])
            res_dict['12'] = 'Prec_no:' + str(Test_Params['precision_All'][3])
            res_dict['13'] = 'Rec_no:' + str(Test_Params['recall_All'][3])
            res_dict['14'] = 'F1_no:' + str(Test_Params['fscore_All'][3])
            res_dict['15'] = 'Prec_spno:' + str(Test_Params['precision_All'][4])
            res_dict['16'] = 'Rec_spno:' + str(Test_Params['recall_All'][4])
            res_dict['17'] = 'F1_spno:' + str(Test_Params['fscore_All'][4])
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
                res_dict['9'] = 'Prec_spmu:' + str(Test_Params['precision_'+str(target_dB)+'dB'][2])
                res_dict['10'] = 'Rec_spmu:' + str(Test_Params['recall_'+str(target_dB)+'dB'][2])
                res_dict['11'] = 'F1_spmu:' + str(Test_Params['fscore_'+str(target_dB)+'dB'][2])                          
                res_dict['12'] = 'Prec_no:' + str(Test_Params['precision_'+str(target_dB)+'dB'][3])
                res_dict['13'] = 'Rec_no:' + str(Test_Params['recall_'+str(target_dB)+'dB'][3])
                res_dict['14'] = 'F1_no:' + str(Test_Params['fscore_'+str(target_dB)+'dB'][3])
                res_dict['15'] = 'Prec_spno:' + str(Test_Params['precision_'+str(target_dB)+'dB'][4])
                res_dict['16'] = 'Rec_spno:' + str(Test_Params['recall_'+str(target_dB)+'dB'][4])
                res_dict['17'] = 'F1_spno:' + str(Test_Params['fscore_'+str(target_dB)+'dB'][4])
                misc.print_results(PARAMS, '', res_dict)
    
        Train_Params = None
        Test_Params = None
    
        if PARAMS['use_GPU']:
            reset_TF_session()
