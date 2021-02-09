#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:47:58 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import datetime
import lib.misc as misc
import configparser
import numpy as np
import lib.feature.preprocessing as preproc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler




def get_harmonic_percussive_score(fv_sp_patches):
    fv_sp_patch_vectors = np.empty([])
    for i in range(np.shape(fv_sp_patches)[0]):
        temp_vec_sp = skew(np.squeeze(fv_sp_patches[i,:]), axis=1) # Along Rows

        temp_vec_sp = np.append(temp_vec_sp, skew(np.squeeze(fv_sp_patches[i,:]), axis=0)) # Along Columns
        temp_vec_sp = np.array(temp_vec_sp, ndmin=2)
        if np.size(fv_sp_patch_vectors)<=1:
            fv_sp_patch_vectors = temp_vec_sp
        else:
            fv_sp_patch_vectors = np.append(fv_sp_patch_vectors, temp_vec_sp, axis=0)
    
    return fv_sp_patch_vectors



def load_data(PARAMS, folder, file_list, target_dB):
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])

    file_list_sp = file_list['speech'].copy()
    file_list_mu = file_list['music'].copy()

    Data_sp = np.empty([], dtype=float)
    Data_mu = np.empty([], dtype=float)
    Data_spmu = np.empty([], dtype=float)
    
    if not os.path.exists(PARAMS['opDir']+'/__data/Data_sp' + '_' + PARAMS['featName'] + PARAMS['HPSS_type'] + '.pkl'):
        for sp_fName in file_list_sp:
            sp_fName_path = folder + '/speech/' + sp_fName
            if not os.path.exists(sp_fName_path):
                continue
            fv_sp = preproc.get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', target_dB)
            fv_sp_patches = preproc.get_feature_patches(PARAMS, fv_sp, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['featName'])
            
            fv_sp_patch_vectors = get_harmonic_percussive_score(fv_sp_patches)
            
            if np.size(Data_sp)<=1:
                Data_sp = fv_sp_patch_vectors
            else:
                Data_sp = np.append(Data_sp, fv_sp_patch_vectors, axis=0)
            print('Speech data: ', np.shape(Data_sp), np.shape(fv_sp_patch_vectors), np.shape(fv_sp_patches))
        if PARAMS['save_flag']:
            misc.save_obj(Data_sp, PARAMS['opDir'], '/__data/Data_sp' + '_' + PARAMS['featName'] + PARAMS['HPSS_type'])
    else:
        Data_sp = misc.load_obj(PARAMS['opDir'], '/__data/Data_sp' + '_' + PARAMS['featName'] + PARAMS['HPSS_type'])
    print('Speech data: ', np.shape(Data_sp))



    if not os.path.exists(PARAMS['opDir']+'/__data/Data_mu' + '_' + PARAMS['featName'] + PARAMS['HPSS_type'] + '.pkl'):
        for mu_fName in file_list_mu:
            mu_fName_path = folder + '/music/' + mu_fName
            if not os.path.exists(mu_fName_path):
                continue
            fv_mu = preproc.get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, target_dB)
            fv_mu_patches = preproc.get_feature_patches(PARAMS, fv_mu, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['featName'])
    
            fv_mu_patch_vectors = get_harmonic_percussive_score(fv_mu_patches)
    
            if np.size(Data_mu)<=1:
                Data_mu = fv_mu_patch_vectors
            else:
                Data_mu = np.append(Data_mu, fv_mu_patch_vectors, axis=0)
            print('Music data: ', np.shape(Data_mu), np.shape(fv_mu_patch_vectors), np.shape(fv_mu_patches))
        if PARAMS['save_flag']:
            misc.save_obj(Data_mu, PARAMS['opDir'], '/__data/Data_mu' + '_' + PARAMS['featName'] + PARAMS['HPSS_type'])
    else:
        Data_mu = misc.load_obj(PARAMS['opDir'], '/__data/Data_mu' + '_' + PARAMS['featName'] + PARAMS['HPSS_type'])
    print('Music data: ', np.shape(Data_mu))


    return Data_mu, Data_sp, Data_spmu




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['draw_tSNE_plots.py']
    PARAMS = {
            'folder': section['folder'], # Folder contaning wav files
            'feature_folder': section['feature_folder'], # Folder containing features
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'CV_folds': int(section['CV_folds']),
            'fold': 0,
            'save_flag': section.getboolean('save_flag'),
            'scale_data': section.getboolean('scale_data'),
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'epochs': int(section['epochs']),
            'batch_size': int(section['batch_size']),
            'Tw': int(section['Tw']),
            'Ts': int(section['Ts']),
            'feature_type': ['Melspectrogram', 'MelHPSS'],
            'HPSS_type_list': ['Harmonic', 'Percussive', 'Both'],
            'HPSS_type': 'Both',
            'n_mels': int(section['n_mels']),
            'silThresh': float(section['silThresh']),
            'classes': {0:'music', 1:'speech', 2:'speech_music'},
            'mixing_dB_range': [-5, -2, -1, 0, 2, 5, 8, 10, 20],
            'task': 'Classification',
            'Nvecs': int(section['Nvecs']),
            'plot_fig': False,
            }

    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()

    for PARAMS['featName'] in PARAMS['feature_type']:
        for PARAMS['HPSS_type'] in PARAMS['HPSS_type_list']:

            PARAMS['feature_opDir'] = PARAMS['feature_folder'] + '/' + PARAMS['folder'].split('/')[-2] + '/' + PARAMS['featName'] + '/'            
            if not os.path.exists(PARAMS['feature_opDir']):
                os.makedirs(PARAMS['feature_opDir'])
    
            cv_file_list = misc.create_CV_folds(PARAMS, PARAMS['folder'], PARAMS['classes'], PARAMS['CV_folds'])
            cv_file_list_test = cv_file_list
            PARAMS['test_folder'] = PARAMS['folder']
            PARAMS['output_folder'] = PARAMS['feature_opDir'] + '/__RESULTS/' + PARAMS['today'] + '/'
    
            PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['featName'] + '_tSNE/'
            if not os.path.exists(PARAMS['opDir']):
                os.makedirs(PARAMS['opDir'])
            
            print('opDir: ', PARAMS['opDir'])
            misc.print_configuration(PARAMS)
    
            if not os.path.exists(PARAMS['opDir']+'/__data/'):
                os.makedirs(PARAMS['opDir']+'/__data/')
    
            if not os.path.exists(PARAMS['opDir']+'/__embeddings/'):
                os.makedirs(PARAMS['opDir']+'/__embeddings/')
                        
            for foldNum in range(1): # range(PARAMS['CV_folds']):
                PARAMS['fold'] = foldNum
                PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
                matFileName = PARAMS['opDir']+'/__embeddings/Data_embedding_'+PARAMS['featName'] + '_' + PARAMS['HPSS_type'] +'.mat'
                
                target_dB = 0
                Data_mu, Data_sp, Data_spmu = load_data(PARAMS, PARAMS['folder'], PARAMS['train_files'], target_dB)
                
                Data = np.append(Data_mu, Data_sp, axis=0)
                # Data = np.append(Data, Data_spmu, axis=0)
                Data = StandardScaler().fit_transform(Data)
                
                Label = np.zeros(np.shape(Data_mu)[0])
                Label = np.append(Label, np.ones(np.shape(Data_sp)[0]))
                # Label = np.append(Label, np.ones(np.shape(Data_spmu)[0])*2)
                
                if PARAMS['HPSS_type']=='Harmonic':
                    Data = Data[:,:21]
                elif PARAMS['HPSS_type']=='Percussive':
                    Data = Data[:,21:]
    
                Label = Label.astype(int)
                print('Data: ', np.shape(Data), PARAMS['HPSS_type'])
                print('Label: ', np.shape(Label), PARAMS['HPSS_type'])
                
                if not os.path.exists(PARAMS['opDir'] + '/Data_embedding.pkl'):             
                    n_components = len(np.unique(Label))
                    print('n_components: ', n_components)
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=30.0,
                        early_exaggeration=12.0,
                        learning_rate=200.0,
                        n_iter=10000,
                        n_iter_without_progress=300,
                        min_grad_norm=1e-07,
                        metric='sqeuclidean',
                        init='pca',
                        verbose=1,
                        random_state=None,
                        # method='exact',
                        method='barnes_hut',
                        angle=0.5,
                        n_jobs=10
                        )                    
                    Data_embedded = tsne.fit_transform(Data)
                    print('Data_embedded: ', np.shape(Data_embedded), np.shape(Data))
                    savemat(matFileName, {'Data':Data_embedded, 'Label':Label})
                else:
                    Data_embedded = misc.load_obj(PARAMS['opDir'], 'Data_embedding')
                    
                if PARAMS['plot_fig']:
                    colors = ['b', 'm', 'k']
                    markers = ['o', 'v', '+']
                    Cl = np.unique(Label)
                    for i in range(len(Cl)):
                        lab = int(Cl[i])
                        idx = np.squeeze(np.where(Label==lab))
                        print(len(idx), lab)
                        plt.scatter(
                            x=Data_embedded[idx,0],
                            y=Data_embedded[idx,1],
                            c=colors[i],
                            edgecolors=colors[i],
                            alpha=1,
                            marker=markers[i],
                            s=50,
                            lw = 1,
                            label=PARAMS['classes'][Cl[i]],
                            )
                        plt.draw()
                    plt.legend(prop={'size': 10})
                    plt.rcParams['font.family'] = 'sans-serif'
                    # plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                    plt.rcParams['font.size'] = '14'
                    plt.show(block=True)
            