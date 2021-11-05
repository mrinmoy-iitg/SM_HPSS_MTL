#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:28:27 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.image import PatchExtractor
import os
from scipy.signal import medfilt
from lib.cython_impl.tools import extract_patches as cextract_patches
from lib.cython_impl.tools import removeSilence as cremoveSilence





def removeSilence(Xin, fs, Tw, Ts, alpha=0.025, beta=0.075):
    '''
    Remove silence regions from audio files

    Parameters
    ----------
    Xin : array
        Audio signal.
    fs : int
        Sampling rate.
    Tw : int
        Short-term frame size (in miliseconds).
    Ts : int
        Short-term frame shift (in miliseconds).
    alpha : float, optional
        Parameter that controls the energy threshold to determine silent 
        frames. The default is 0.025.
    beta : float, optional
        Parameter that determines the minimum duration of a silent 
        segment (in seconds). The default is 0.075.

    Returns
    -------
    Xin_silrem : array
        Silence removed audio signal.
    sample_silMarker : array
        Silence/Non-silence annotation for every sample in the original audio
        signal.
    frame_silMarker : array
        Silence/Non-silence annotation for every short-term frame in the 
        original audio signal.
    totalSilDuration : int
        Duration of all silence segments.

    '''
    frameSize = int((Tw*fs)/1000) # Frame size in number of samples
    frameShift = int((Ts*fs)/1000) # Frame shift in number of samples
    
    Rmse = librosa.feature.rms(y=Xin, frame_length=frameSize, hop_length=frameShift)
    energy = Rmse[0,:] #pow(Rmse,2)
    energyThresh = alpha*np.max(energy) # TEST WITH MEAN

    frame_silMarker = energy
    frame_silMarker[frame_silMarker < energyThresh] = 0
    frame_silMarker[frame_silMarker >= energyThresh] = 1
    silences = np.empty([])
    totalSilDuration = 0
    
    # Removing spurious detections
    frame_silMarker = medfilt(frame_silMarker.flatten(), 5)
    frame_silMarker = (frame_silMarker>0.5).astype(int)

    sample_silMarker = np.ones(len(Xin))
    i=0
    while i<len(frame_silMarker):
        while frame_silMarker[i]==1:
            if i == len(frame_silMarker)-1:
                break
            i = i + 1
        j = i
        while frame_silMarker[j]==0:
            if j == len(frame_silMarker)-1:
                break
            j = j + 1
        k = np.max([frameShift*(i-1)+frameSize, 1])
        l = np.min([frameShift*(j-1)+frameSize,len(Xin)]);
        
        # Only silence segments of durations greater than given beta
        # (e.g. 100ms) are removed
        if (l-k)/fs > beta:
            sample_silMarker[k:l] = 0
            if np.size(silences)<=1:
                silences = np.array([k,l], ndmin=2)
            else:
                silences = np.append(silences, np.array([k,l], ndmin=2),0)
            totalSilDuration += (l-k)/fs
        i = j + 1
    
    if np.size(silences)>1:
        Xin_silrem = np.empty([]) #Xin
        for i in range(np.shape(silences)[0]):
            if i==0:
                Xin_silrem = Xin[:silences[i,0]]
            else:
                Xin_silrem = np.append(Xin_silrem, Xin[silences[i-1,1]:silences[i,0]])
    else:
        Xin_silrem = Xin
        
    return Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration




def normalize_signal(Xin):
    '''
    Normalize an audio signal by subtracting mean and dividing my the maximum
    amplitude

    Parameters
    ----------
    Xin : array
        Audio signal.

    Returns
    -------
    Xin : array
        Normalized audio signal.

    '''
    Xin = Xin - np.mean(Xin)
    Xin = Xin / np.max(np.abs(Xin))
    return Xin




def get_feature_patches(PARAMS, FV, patch_size, patch_shift, featName):
    # FV should be of the shape (nFeatures, nFrames)
    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)        

    if featName=='Spec':
        if not PARAMS['frame_level_scaling']:
            FV = FV.T
            FV = StandardScaler(copy=False).fit_transform(FV)
            FV = FV.T
        patches = cextract_patches(FV, np.shape(FV), patch_size, patch_shift)
        if not 'Lemaire_et_al' in PARAMS['Model']:
            patches = np.expand_dims(patches, axis=3)

    if featName=='LogSpec':
        if not PARAMS['frame_level_scaling']:
            FV = FV.T
            FV = StandardScaler(copy=False).fit_transform(FV)
            FV = FV.T
        patches = cextract_patches(FV, np.shape(FV), patch_size, patch_shift)
        if not 'Lemaire_et_al' in PARAMS['Model']:
            patches = np.expand_dims(patches, axis=3)
    
    elif featName=='MelSpec':
        if not PARAMS['frame_level_scaling']:
            FV = FV.T
            FV = StandardScaler(copy=False).fit_transform(FV)
            FV = FV.T
        patches = cextract_patches(FV, np.shape(FV), patch_size, patch_shift)
        if not 'Lemaire_et_al' in PARAMS['Model']:
            patches = np.expand_dims(patches, axis=3)

    elif featName=='LogMelSpec':
        if not PARAMS['frame_level_scaling']:
            FV = FV.T
            FV = StandardScaler(copy=False).fit_transform(FV)
            FV = FV.T
        patches = cextract_patches(FV, np.shape(FV), patch_size, patch_shift)
        if not 'Lemaire_et_al' in PARAMS['Model']:
            patches = np.expand_dims(patches, axis=3)
        
    elif featName.startswith('MelHarm') or featName.startswith('MelPerc'):
        if (featName=='MelHarmSpec') or (featName=='MelHarmPercSpec'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            if not PARAMS['frame_level_scaling']:
                FV_H = FV_H.T
                FV_H = StandardScaler(copy=False).fit_transform(FV_H)
                FV_H = FV_H.T
            patches_H = cextract_patches(FV_H, np.shape(FV_H), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_H = np.expand_dims(patches_H, axis=3)

        if (featName=='MelPercSpec') or (featName=='MelHarmPercSpec'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            if not PARAMS['frame_level_scaling']:
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

    elif featName.startswith('LogMelHarm') or featName.startswith('LogMelPerc'):
        if (featName=='LogMelHarmSpec') or (featName=='LogMelHarmPercSpec'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            if not PARAMS['frame_level_scaling']:
                FV_H = FV_H.T
                FV_H = StandardScaler(copy=False).fit_transform(FV_H)
                FV_H = FV_H.T
            patches_H = cextract_patches(FV_H, np.shape(FV_H), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_H = np.expand_dims(patches_H, axis=3)

        if (featName=='LogMelPercSpec') or (featName=='LogMelHarmPercSpec'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            if not PARAMS['frame_level_scaling']:
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

    elif featName.startswith('Harm') or featName.startswith('Perc'):
        if (featName=='HarmSpec') or (featName=='HarmPercSpec'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            if not PARAMS['frame_level_scaling']:
                FV_H = FV_H.T
                FV_H = StandardScaler(copy=False).fit_transform(FV_H)
                FV_H = FV_H.T
            patches_H = cextract_patches(FV_H, np.shape(FV_H), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_H = np.expand_dims(patches_H, axis=3)

        if (featName=='PercSpec') or (featName=='HarmPercSpec'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            if not PARAMS['frame_level_scaling']:
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

    elif featName.startswith('LogHarm') or featName.startswith('LogPerc'):
        if (featName=='LogHarmSpec') or (featName=='LogHarmPercSpec'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            if not PARAMS['frame_level_scaling']:
                FV_H = FV_H.T
                FV_H = StandardScaler(copy=False).fit_transform(FV_H)
                FV_H = FV_H.T
            patches_H = cextract_patches(FV_H, np.shape(FV_H), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_H = np.expand_dims(patches_H, axis=3)

        if (featName=='LogPercSpec') or (featName=='LogHarmPercSpec'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            if not PARAMS['frame_level_scaling']:
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




def mix_signals(Xin_sp, Xin_mu, target_dB):
    sig_sp_len = len(Xin_sp)
    sig_mu_len = len(Xin_mu)
    Xin_mu_temp = Xin_mu.copy()
    # Music length must be equal to or more than the speech length
    while sig_mu_len<sig_sp_len:
        Xin_mu_temp = np.append(Xin_mu_temp, Xin_mu)
        sig_mu_len = len(Xin_mu_temp)
    Xin_mu = Xin_mu_temp.copy()
    sig_mu_len = len(Xin_mu)
    common_len = np.min([sig_sp_len, sig_mu_len])
    Xin_sp = Xin_sp[:common_len]
    Xin_mu = Xin_mu[:common_len]
    
    sig_sp_energy = np.sum(np.power(Xin_sp,2))/len(Xin_sp)
    sig_mu_energy = np.sum(np.power(Xin_mu,2))/len(Xin_mu)
    
    req_sig_mu_energy = sig_sp_energy/np.power(10,(target_dB/10))
    sig_mu_mult_fact = np.sqrt(req_sig_mu_energy/sig_mu_energy)
    
    sig_sp_mult_fact = 1
    mult_fact_sum = sig_mu_mult_fact+sig_sp_mult_fact
    sig_mu_mult_fact /= mult_fact_sum
    sig_sp_mult_fact /= mult_fact_sum
    
    Xin_mix = sig_sp_mult_fact*Xin_sp + sig_mu_mult_fact*Xin_mu
    Xin_mix = normalize_signal(Xin_mix)
    
    return Xin_mix




def load_and_preprocess_signal(fName, Tw, Ts):
    Xin, fs = librosa.core.load(fName, mono=True, sr=16000)
    Xin = normalize_signal(Xin)
    # Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = removeSilence(Xin, fs, Tw, Ts)

    frameSize = int((Tw*fs)/1000) # Frame size in number of samples
    frameShift = int((Ts*fs)/1000) # Frame shift in number of samples
    Rmse = librosa.feature.rms(y=Xin, frame_length=frameSize, hop_length=frameShift)
    energy = Rmse[0,:] #pow(Rmse,2)
    Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = cremoveSilence(Xin, len(Xin), energy, np.shape(Rmse)[1], fs, Tw, Ts)
    
    if np.size(Xin_silrem)<=1:
        Xin = Xin_silrem
    Xin = Xin_silrem.copy()

    if len(Xin)/fs < 0.1:
        while len(Xin)/fs<0.1:
            Xin = np.append(Xin, Xin)
    Xin_norm = normalize_signal(Xin)
    
    return Xin_norm, fs




def get_featuregram(PARAMS, classname, feature_opDir, fName_path_sp, fName_path_mu, target_dB, n_fft, n_mels, featName, save_feat=True):
    if (fName_path_sp!='') and (fName_path_mu!=''): # speech_music
        fName = fName_path_sp.split('/')[-1].split('.')[0]+'_'+fName_path_mu.split('/')[-1].split('.')[0]+'_'+str(target_dB)+'dB'
    elif fName_path_sp!='': # speech
        fName = fName_path_sp.split('/')[-1].split('.')[0]
    elif fName_path_mu!='': # music
        fName = fName_path_mu.split('/')[-1].split('.')[0]

    if not os.path.exists(feature_opDir+'/'+classname+'/'+fName+'.npy'):
        if classname=='speech_music':
            Xin_sp, fs = load_and_preprocess_signal(fName_path_sp, PARAMS['Tw'], PARAMS['Ts'])
            Xin_mu, fs = load_and_preprocess_signal(fName_path_mu, PARAMS['Tw'], PARAMS['Ts'])
            Xin = mix_signals(Xin_sp, Xin_mu, target_dB)
    
        elif classname=='speech':
            Xin, fs = load_and_preprocess_signal(fName_path_sp, PARAMS['Tw'], PARAMS['Ts'])
    
        elif classname=='music':
            Xin, fs = load_and_preprocess_signal(fName_path_mu, PARAMS['Tw'], PARAMS['Ts'])

        elif classname=='muspeak':
            Xin, fs = load_and_preprocess_signal(fName_path_sp, PARAMS['Tw'], PARAMS['Ts'])

        if featName=='Spec':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            fv = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            fv = fv.astype(np.float32)

        if featName=='LogSpec':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            fv = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            fv = librosa.core.power_to_db(fv**2)
            fv = fv.astype(np.float32)

        elif featName=='MelSpec':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            fv = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False, n_mels=n_mels)
            fv = fv.astype(np.float32)

        elif featName=='LogMelSpec':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            fv = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False, n_mels=n_mels)
            fv = librosa.core.power_to_db(fv**2)
            fv = fv.astype(np.float32)

        elif featName.startswith('MelHarm') or featName.startswith('MelPerc'):
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            Spec = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            H_Spec, P_Spec = librosa.decompose.hpss(S=Spec, kernel_size=(PARAMS['l_harm'][PARAMS['Model']], PARAMS['l_perc'][PARAMS['Model']]))
            fv_H = librosa.feature.melspectrogram(S=H_Spec, n_mels=n_mels)
            fv_P = librosa.feature.melspectrogram(S=P_Spec, n_mels=n_mels)
            fv = np.append(fv_H, fv_P, axis=0) 
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

        elif featName.startswith('Harm') or featName.startswith('Perc'):
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            Spec = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            H_Spec, P_Spec = librosa.decompose.hpss(S=Spec, kernel_size=(PARAMS['l_harm'][PARAMS['Model']], PARAMS['l_perc'][PARAMS['Model']]))
            fv_H = H_Spec
            fv_P = P_Spec
            fv = np.append(fv_H, fv_P, axis=0) 
            fv = fv.astype(np.float32)

        elif featName.startswith('LogHarm') or featName.startswith('LogPerc'):
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            Spec = np.abs(librosa.core.stft(y=Xin, n_fft=n_fft, win_length=frameSize, hop_length=frameShift, center=False))
            H_Spec, P_Spec = librosa.decompose.hpss(S=Spec, kernel_size=(PARAMS['l_harm'][PARAMS['Model']], PARAMS['l_perc'][PARAMS['Model']]))
            fv_H = librosa.core.power_to_db(H_Spec**2)
            fv_P = librosa.core.power_to_db(P_Spec**2)
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
            fv = np.load(feature_opDir+'/'+classname+'/'+fName+'.npy', allow_pickle=True)
    
    return fv



def get_data_stats(PARAMS, files):
    classes = PARAMS['classes']
    folder = PARAMS['feature_opDir']
    featName = PARAMS['featName'][PARAMS['Model']]
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]
    
    mean_mu = np.empty([], dtype=np.float128)
    nMuFrames = 0
    mean_sp = np.empty([], dtype=np.float128)
    nSpFrames = 0
    mean_spmu = np.empty([], dtype=np.float128)
    nSpMuFrames = 0
    
    for clNum in classes.keys():
        class_name = classes[clNum]
        if class_name=='speech_music':
            file_list = files['speech+music']
        else:
            file_list = files[class_name]
        print(class_name, len(file_list))
        for fl in file_list:
            if class_name=='speech_music':
                fName_mu = fl['music'].split('/')[-1].split('.')[0]
                fName_sp = fl['speech'].split('/')[-1].split('.')[0]
                fName = folder + '/' + class_name + '/' + fName_sp + '_' + fName_mu + '_' + str(fl['SMR']) + 'dB.npy'
            else:
                fName = folder + '/' + class_name + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
            if not os.path.exists(fName):
                if class_name=='speech_music':
                    sp_fName = fl['speech']
                    sp_fName_path = PARAMS['folder'] + '/speech/' + sp_fName
                    mu_fName = fl['music']
                    mu_fName_path = PARAMS['folder'] + '/music/' + mu_fName
                    target_dB = fl['SMR']
                    FV = get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], sp_fName_path, mu_fName_path, target_dB, n_fft, n_mels, featName)
                elif class_name=='music':
                    mu_fName_path = PARAMS['folder'] + '/music/' + fl.split('.')[0] + '.wav'
                    FV = get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, -1, n_fft, n_mels, featName)
                elif class_name=='speech':
                    sp_fName_path = PARAMS['folder'] + '/speech/' + fl.split('.')[0] + '.wav'
                    FV = get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', -1, n_fft, n_mels, featName)
                print('FV: ', np.shape(FV), fl)
            else:
                FV = np.load(fName, allow_pickle=True)
            FV = FV[~np.isnan(FV).any(axis=1), :]
            FV = FV[~np.isinf(FV).any(axis=1), :]
            FV = FV.T
            nFrames = np.shape(FV)[0]
            
            if class_name=='music':
                nMuFrames += np.shape(FV)[0]
                if np.size(mean_mu)<=1:
                    mean_mu = np.sum(FV, axis=0)                
                else:
                    mean_mu = np.add(mean_mu, np.sum(FV, axis=0))
            elif class_name=='speech':
                nSpFrames += np.shape(FV)[0]
                if np.size(mean_sp)<=1:
                    mean_sp = np.sum(FV, axis=0)                
                else:
                    mean_sp = np.add(mean_sp, np.sum(FV, axis=0))
            elif class_name=='speech_music':
                nSpMuFrames += np.shape(FV)[0]
                if np.size(mean_spmu)<=1:
                    mean_spmu = np.sum(FV, axis=0)                
                else:
                    mean_spmu = np.add(mean_spmu, np.sum(FV, axis=0))
    mean_mu /= nMuFrames+1e-10
    mean_sp /= nSpFrames+1e-10
    mean_spmu /= nSpMuFrames+1e-10
    if len(classes)==2:
        overall_mean = np.add(mean_mu, mean_sp)/2
    elif len(classes)==3:        
        overall_mean = np.add(np.add(mean_mu, mean_sp), mean_spmu)/3
    print('Overall mean: ', np.shape(overall_mean), np.round(overall_mean, 4))

    stdev = np.empty([], dtype=np.float128)
    nFrames = 0
    for clNum in classes.keys():
        class_name = classes[clNum]
        if class_name=='speech_music':
            file_list = files['speech+music']
        else:
            file_list = files[class_name]
        for fl in file_list:
            if class_name=='speech_music':
                fName_mu = fl['music'].split('/')[-1].split('.')[0]
                fName_sp = fl['speech'].split('/')[-1].split('.')[0]
                fName = folder + '/' + class_name + '/' + fName_sp + '_' + fName_mu + '_' + str(fl['SMR']) + 'dB.npy'
            else:
                fName = folder + '/' + class_name + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
            if not os.path.exists(fName):
                if class_name=='speech_music':
                    sp_fName = fl['speech']
                    sp_fName_path = PARAMS['folder'] + '/speech/' + sp_fName
                    mu_fName = fl['music']
                    mu_fName_path = PARAMS['folder'] + '/music/' + mu_fName
                    target_dB = fl['SMR']
                    FV = get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], sp_fName_path, mu_fName_path, target_dB, n_fft, n_mels, featName)
                elif class_name=='music':
                    mu_fName_path = PARAMS['folder'] + '/music/' + fl.split('.')[0] + '.wav'
                    FV = get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, -1, n_fft, n_mels, featName)
                elif class_name=='speech':
                    sp_fName_path = PARAMS['folder'] + '/speech/' + fl.split('.')[0] + '.wav'
                    FV = get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', -1, n_fft, n_mels, featName)
                print('FV: ', np.shape(FV), os.path.exists(fName), fName)
            else:
                FV = np.load(fName, allow_pickle=True)
            FV = FV[~np.isnan(FV).any(axis=1), :]
            FV = FV[~np.isinf(FV).any(axis=1), :]
            FV = FV.T # nFrames x nDim
            nFrames += np.shape(FV)[0]
            mean_arr = np.repeat(np.array(overall_mean, ndmin=2), np.shape(FV)[0], axis=0)
            if np.size(stdev)<=1:
                stdev = np.sum(np.power(np.subtract(FV, mean_arr), 2), axis=0)
            else:
                stdev = np.add(stdev, np.sum(np.power(np.subtract(FV, mean_arr),2), axis=0))
    
    print(stdev)
    stdev /= (nFrames-1)
    stdev = np.sqrt(stdev)
    print('Stdev: ', np.shape(stdev), np.round(stdev, 4))
        
    return overall_mean.astype(np.float32), stdev.astype(np.float32), nMuFrames, nSpFrames, nSpMuFrames



def scale_data(FV, mean, stdev):
    '''
    Mean-variance scaling of the data.

    Parameters
    ----------
    FV : array
        Feature Vector.
    mean : array
        Frame mean.
    stdev : array
        Frame standard-deviation.

    Returns
    -------
    FV_scaled : TYPE
        DESCRIPTION.

    '''
    M = np.repeat(np.array(mean, ndmin=2).T, np.shape(FV)[1], axis=1)
    S = np.repeat(np.array(stdev, ndmin=2).T, np.shape(FV)[1], axis=1)
    FV_scaled = FV.copy()
    FV_scaled = np.subtract(FV_scaled, M)
    FV_scaled = np.divide(FV_scaled, S)
    return FV_scaled
    