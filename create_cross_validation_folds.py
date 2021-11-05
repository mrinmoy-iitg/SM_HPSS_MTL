#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:41:05 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import librosa
import numpy as np
import lib.misc as misc
import csv


def duration_format(N):
    '''
    Conversion of number of seconds into HH-mm-ss format
    
    Parameters
    ----------
    N : int
        Number of seconds.

    Returns
    -------
    H : int
        Hours.
    M : int
        Minutes.
    S : int
        Seconds.

    Created: 24 July, 2021
    '''
    H = int(N/3600)
    N = N%3600
    M = int(N/60)
    S = N%60
    return H,M,S


def calculate_dataset_size(folder, classes):
    '''
    This function calculates the size of the dataset indicated by the path in
    "folder"

    Parameters
    ----------
    folder : str
        Path to dataset.
    classes : dict
        Key-value pairs of class-labels and names.

    Returns
    -------
    duration : dict
        Key-value pairs of classnames and their durations.
    filewise_duration : dict
        Key-value pairs of filenames and their durations.

    Created: 24 July, 2021
    '''
    
    duration = {classes[key]:0 for key in classes}
    filewise_duration = {classes[key]:{} for key in classes}
    for classname in duration:
        class_data_path = folder + '/' + classname + '/'
        files = [fl.split('/')[-1] for fl in librosa.util.find_files(class_data_path, ext=['wav'])]
        count = 0
        progress_mark = ['-', '\\', '|', '/']
        for fl in files:
            fName = class_data_path + fl
            Xin, fs = librosa.core.load(fName, mono=True, sr=None)
            file_duration = len(Xin)/fs
            duration[classname] += file_duration
            filewise_duration[classname][fl] = file_duration
            H,M,S = duration_format(int(duration[classname]))
            print(classname, progress_mark[count%4], H, 'hr', M, 'min', S, 'sec', end='\r', flush=True)
            count += 1
        print('\n')
    return duration, filewise_duration



def get_annotations(folder, data_path):
    '''
    Read annotations provided with the dataset for speech and music 
    files

    Parameters
    ----------
    folder : str
        Path to the annotations.

    Returns
    -------
    annotations_mu : dict
        A dict containing the annotations for each music file.
    annotations_sp : TYPE
        A dict containing the annotations for each speech file.

    Created: 25 July, 2021
    '''
    annotations_mu = {}
    try:
        with open(folder+'/music.csv', newline='\n') as csvfile:
            annotreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_count = 0
            for row in annotreader:
                if row==[]:
                    continue
                annotations_mu[row_count] = row
                row_count += 1
    except:
        print('No file-level annotations available for the MUSIC data')
        files_mu = [fl.split('/')[-1].split('.')[0] for fl in librosa.util.find_files(data_path+'/music/', ext=['wav'])]
        fl_count = 0
        for fl in files_mu:
            annotations_mu[fl_count] = [fl, 'no_annot']
            fl_count += 1
        
    annotations_sp = {}
    try:
        with open(folder+'/speech.csv', newline='\n') as csvfile:
            annotreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_count = 0
            for row in annotreader:
                if row==[]:
                    continue
                annotations_sp[row_count] = row
                row_count += 1
    except:
        print('No file-level annotations available for the SPEECH data')
        files_sp = [fl.split('/')[-1].split('.')[0] for fl in librosa.util.find_files(data_path+'/speech/', ext=['wav'])]
        fl_count = 0
        for fl in files_sp:
            annotations_sp[fl_count] = [fl, 'no_annot']
            fl_count += 1

    return annotations_mu, annotations_sp



def create_CV_folds(folder, dataset_name, classes, CV, total_duration, filewise_duration, mixing_dB_range):
    '''
    This function divides the audio files in the speech and music classes of 
    the dataset into non-overlapping folds. Additionally, each fold of the 
    speech+music class is created by randomly selecting one speech and one 
    music file from their respective folds. The number of file combinations 
    in each fold of the speech+music class is made equal to the largest 
    constituent class.

    Parameters
    ----------
    folder : str
        Path to the dataset.
    dataset_name : str
        Name of the dataset.
    classes : dict
        A dict containing class labels and names in key-value pairs.
    CV : int
        Number of cross-validation folds to be created.
    total_duration : dict
        A dict containing the total durations of speech and music classes as 
        key-value pairs.
    filewise_duration : dict
        A dict containing the filenames and their respective durations as 
        key-value pairs for speech and music classes.
    mixing_dB_range: list
        List of all SMR in dB for mixing the speech and music files

    Returns
    -------
    cv_file_list : TYPE
        Key-value pairs of classes and the list of files divided into 
        non-overlapping folds.
    
    Created: 24 July, 2021
    '''
    annotations_mu, annotations_sp = get_annotations('./annotations/'+dataset_name+'/', folder)

    cv_file_list = {'CV_folds': CV, 'dataset_name': dataset_name, 'music':{'fold0':[], 'fold1':[], 'fold2':[]}, 'speech':{'fold0':[], 'fold1':[], 'fold2':[]}}
    music_annot = {}
    for row in annotations_mu.keys():
        fName = annotations_mu[row][0]
        if os.path.exists(folder+'/music/'+fName+'.wav'):
            g = annotations_mu[row][1]
            if not g in music_annot.keys():
                music_annot[g] = {'fold0':[], 'fold1':[], 'fold2':[], 'last_added_fold':0}
                music_annot[g]['fold0'].append(fName+'.wav')
            else:
                music_annot[g]['last_added_fold'] += 1
                if music_annot[g]['last_added_fold']==3:
                    music_annot[g]['last_added_fold'] = 0
                music_annot[g]['fold'+str(music_annot[g]['last_added_fold'])].append(fName+'.wav')
            cv_file_list['music']['fold'+str(music_annot[g]['last_added_fold'])].append(fName+'.wav')
        
    speech_annot = {}
    for row in annotations_sp.keys():
        fName = annotations_sp[row][0]
        if os.path.exists(folder+'/speech/'+fName+'.wav'):
            g = annotations_sp[row][1]
            if not g in speech_annot.keys():
                speech_annot[g] = {'fold0':[], 'fold1':[], 'fold2':[], 'last_added_fold':0}
                speech_annot[g]['fold0'].append(fName+'.wav')
            else:
                speech_annot[g]['last_added_fold'] += 1
                if speech_annot[g]['last_added_fold']==3:
                    speech_annot[g]['last_added_fold'] = 0
                speech_annot[g]['fold'+str(speech_annot[g]['last_added_fold'])].append(fName+'.wav')
            cv_file_list['speech']['fold'+str(speech_annot[g]['last_added_fold'])].append(fName+'.wav')

    for clNum in classes.keys():
        path = folder + '/' + classes[clNum] + '/'
        files = [fl.split('/')[-1] for fl in librosa.util.find_files(path, ext=['wav'])]
        np.random.shuffle(files)            
        for cv_num in range(CV):
            fold_duration = 0
            for fl in cv_file_list[classes[clNum]]['fold'+str(cv_num)]:
                fold_duration += filewise_duration[classes[clNum]][fl]

    print(cv_file_list['music'])
    print(cv_file_list['speech'])
                                
    cv_file_list['speech+music'] = {}
    for cv_num in range(CV):
        cv_file_list['speech+music']['fold'+str(cv_num)] = []
        files_sp = cv_file_list['speech']['fold'+str(cv_num)].copy()
        np.random.shuffle(files_sp)
        files_mu = cv_file_list['music']['fold'+str(cv_num)].copy()
        np.random.shuffle(files_mu)
        db_idx = 0
        for nFiles in range(np.max([len(files_sp), len(files_mu)])):
            if len(files_sp)==0:
                files_sp = cv_file_list['speech']['fold'+str(cv_num)].copy()
                np.random.shuffle(files_sp)
            fl_sp = files_sp.pop()
            if len(files_mu)==0:
                files_mu = cv_file_list['music']['fold'+str(cv_num)].copy()
                np.random.shuffle(files_mu)                
            fl_mu = files_mu.pop()
            cv_file_list['speech+music']['fold'+str(cv_num)].append({'speech':fl_sp, 'music':fl_mu, 'SMR':mixing_dB_range[db_idx]})
            db_idx += 1
            if db_idx>=len(mixing_dB_range):
                db_idx = 0

    cv_file_list['filewise_duration'] = filewise_duration
    cv_file_list['total_duration'] = total_duration
    cv_file_list['total_duration']['speech+music'] = np.max([val for val in total_duration.values()])
    for key in cv_file_list['total_duration'].keys():
        cv_file_list['total_duration'][key] /= 3600 # in Hours

    print('total_duration: ', total_duration)
    dataset_size = 0 # in Hours
    for classname in cv_file_list['total_duration'].keys():
        dataset_size += cv_file_list['total_duration'][classname]
    print('Dataset size: ', dataset_size, 'Hrs', total_duration)
    cv_file_list['dataset_size'] = dataset_size
    
    return cv_file_list, music_annot, speech_annot



def print_cv_info(cv_file_list, opDir, CV):
    '''
    Print the DESCRIPTIONresult of cross-validation fold distribution of files generated 
    for the dataset

    Parameters
    ----------
    cv_file_list : dict
        Class-wise key-value pairs of files in different cross-validation 
        folds.
    opDir : str
        Path to store the result files.
    CV : int
        Number of cross-validation folds.

    Returns
    -------
    None.

    Created: 24 July, 2021
    '''
    fid = open(opDir+'/details.txt', 'w+', encoding='utf8')
    for key in cv_file_list.keys():
        fid.write(key + ': ' + str(cv_file_list[key]) +'\n\n\n')
    fid.close()

    for fold in range(CV):
        fid = open(opDir+'/fold' + str(fold) + '.csv', 'w+', encoding='utf8')
        sp_files = cv_file_list['speech']['fold'+str(fold)]
        mu_files = cv_file_list['music']['fold'+str(fold)]
        spmu_files = cv_file_list['speech+music']['fold'+str(fold)]
        maxFiles = np.max([len(sp_files), len(mu_files), len(spmu_files)])
        fid.write('music,speech,speech+music\n')
        for i in range(maxFiles):
            if len(sp_files)>i:
                sp_fName = sp_files[i]
            else:
                sp_fName = ''
            if len(mu_files)>i:
                mu_fName = mu_files[i]
            else:
                mu_fName = ''
            if len(spmu_files)>i:
                spmu_fName = spmu_files[i]['speech']+'+'+spmu_files[i]['music']+';SMR='+str(spmu_files[i]['SMR'])+'dB'
            else:
                spmu_fName = ''
            row = mu_fName+','+sp_fName+','+spmu_fName+'\n'
            fid.write(row)
        fid.close()    



if __name__ == '__main__':
    folder = '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/musan/'
    # folder = '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Movie_Music_Speech_Noise_Mix_Corpus_Movie-MUSNOMIX/MUSNOMIX_WAV/'
    classes = {0:'music', 1:'speech'}
    CV = 3
    dataset_name = list(filter(None,folder.split('/')))[-1]
    opDir = './cross_validation_info/' + dataset_name + '/'
    if not os.path.exists(opDir):
        os.makedirs(opDir)
    mixing_dB_range = list(range(-5,21)) # -5dB to 20dB mixing SMR levels
            
    if not os.path.exists(opDir+'/Dataset_Duration.pkl'):
        total_duration, filewise_duration = calculate_dataset_size(folder, classes)
        misc.save_obj({'total_duration':total_duration, 'filewise_duration':filewise_duration}, opDir, 'Dataset_Duration')
    else:
        total_duration = misc.load_obj(opDir, 'Dataset_Duration')['total_duration']
        filewise_duration = misc.load_obj(opDir, 'Dataset_Duration')['filewise_duration']    
    
    if not os.path.exists(opDir+'/cv_file_list.pkl'):
        cv_file_list, music_annot, speech_annot = create_CV_folds(folder, dataset_name, classes, CV, total_duration, filewise_duration, mixing_dB_range)
        misc.save_obj(cv_file_list, opDir, 'cv_file_list')
        misc.save_obj(music_annot, opDir, 'music_annot')
        misc.save_obj(speech_annot, opDir, 'speech_annot')
        print('CV folds created')
    else:
        cv_file_list = misc.load_obj(opDir, 'cv_file_list')
        music_annot = misc.load_obj(opDir, 'music_annot')
        speech_annot = misc.load_obj(opDir, 'speech_annot')
        print('CV folds loaded')
    
    print_cv_info(cv_file_list, opDir, CV)