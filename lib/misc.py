#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:55:09 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json



def save_obj(obj, folder, name):
    with open(folder+'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def load_obj(folder, name):
    with open(folder+'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



def scale_data(train_data, test_data):
    print('Scale data func: train_data=', np.shape(train_data))
    std_scale = StandardScaler().fit(train_data)
    train_data_scaled = std_scale.transform(train_data)
    test_data_scaled = std_scale.transform(test_data)

    return train_data_scaled, test_data_scaled


    
def preprocess_data(PARAMS, train_data, train_label, test_data):
    if PARAMS['data_balancing']:
        from imblearn.combine import SMOTEENN
        print('Unbalanced data: ', np.shape(train_data))
        # Over and under sampling
        smote_enn = SMOTEENN(sampling_strategy=1.0)
        train_data, train_label = smote_enn.fit_resample(train_data, train_label)
        print('Balanced data: ', np.shape(train_data))
    
    if PARAMS['scale_data']:
        train_data, test_data = scale_data(train_data, test_data)
    
    return train_data, train_label, test_data
    



def get_train_test_files(cv_file_list, cv_file_list_test, numCV, foldNum):
    train_files = {}
    test_files = {}
    for class_name in cv_file_list.keys():
        train_files[class_name] = []
        test_files[class_name] = []
        for i in range(numCV):
            files = cv_file_list[class_name]['fold'+str(i)]
            files_test = cv_file_list_test[class_name]['fold'+str(i)]
            if foldNum==i:
                test_files[class_name].extend(files_test)
            else:
                train_files[class_name].extend(files)
    
    return train_files, test_files

    
    
    
def load_data_from_files(classes, folder, featName, files):
    label = []
    data = np.empty([])
    for clNum in classes.keys():
        for fl in files[classes[clNum]]:
            FV = np.load(folder + '/' + featName + '/' + classes[clNum] + '/' + fl, allow_pickle=True)
            if np.size(data)<=1:
                data = FV
            else:
                # print('data: ', np.shape(data), np.shape(FV))
                data = np.append(data, FV, 0)
            label.extend([clNum]*np.shape(FV)[0])
    label = np.array(label, ndmin=2).T
    return data, label



def getPerformance(PtdLabels, GroundTruths, labels):
    ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels)
    # fscore = f1_score(GroundTruths, PtdLabels, average=None, labels=[0,1]).tolist()    
    precision, recall, fscore, support = precision_recall_fscore_support(y_true=GroundTruths, y_pred=PtdLabels, beta=1.0, average=None, labels=labels)
    precision = np.round(precision,4)
    recall = np.round(recall,4)
    fscore = np.round(fscore,4)

    return ConfMat, precision, recall, fscore





def print_results(PARAMS, fName_suffix, res_dict):
    if fName_suffix!='':
        opFile = PARAMS['opDir'] + '/Performance_' + fName_suffix + '.csv'
    else:
        opFile = PARAMS['opDir'] + '/Performance.csv'
        
    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = 'fold'
    values = str(PARAMS['fold'])
    for i in range(len(res_dict)):
        heading = heading + '\t' + np.squeeze(res_dict[str(i)]).tolist().split(':')[0]
        values = values + '\t' + np.squeeze(res_dict[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




def print_configuration(PARAMS):
    opFile = PARAMS['opDir'] + '/Configuration.csv'
    fid = open(opFile, 'a+', encoding = 'utf-8')
    PARAM_keys = [key for key in PARAMS.keys()]
    for i in range(len(PARAM_keys)):
        if PARAM_keys[i]=='GPU_session':
            continue
        # print('PARAMS key: ', PARAM_keys[i])
        try:
            fid.write(PARAM_keys[i] + '\t')
            fid.write(json.dumps(PARAMS[PARAM_keys[i]]))
            fid.write('\n')
        except:
            fid.write(PARAM_keys[i] + '\tERROR\n')
            
    fid.close()
    



def print_analysis(opFile, results):
    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = ''
    values = ''
    for i in range(len(results)):
        if heading=='':
            heading = np.squeeze(results[str(i)]).tolist().split(':')[0]
            values = np.squeeze(results[str(i)]).tolist().split(':')[1]
        else:
            heading += '\t' + np.squeeze(results[str(i)]).tolist().split(':')[0]
            values += '\t' + np.squeeze(results[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()
    
    
def print_model_summary(arch_file, model):
    stringlist = ['']
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    with open(arch_file, 'w+', encoding='utf8') as f:
        f.write(short_model_summary)
