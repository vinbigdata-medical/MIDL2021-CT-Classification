from tqdm import tqdm 
import pandas as pd
import os
import torch
import torch.nn as nn
from shutil import copyfile

from sklearn.metrics import multilabel_confusion_matrix, classification_report

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

def record_output(cfg, mode:str, log_dir:str, 
                targets:list, 
                predictions: list, 
                filenames:list):
    """Save all wrongly predicted in 'log_dir' folder, 
    along with the confusion matrix. This can only be used for
    2D slice-level prediction

    Args:
        cfg (CfgNode): Config object containing running configuration
        mode (str): Model running mode (valid/test)
        log_dir (str): Directory to save images and the confusion matrix
        targets (list): List of groundtruth labels  (size = N). Ex: [0,1,2,1,1,0]
        predictions (list): List of predicted labels (size = N)
        filenames (list): List of filenames (size = N). This is used when naming
                    wrongly predicted images.
    """
    # PREPARE DIRECTORY
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, cfg.NAME)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Record wrongly predicted sample as df
    data = {'StudyID': [], 'Filename': [], 'slice_no': [], 
            'Prediction': [], 'Target': []}
    for target, pred, filename in zip(targets, predictions, filenames):
        if target != pred:
            data['StudyID'].append(filename.split('/')[-2])
            data['Filename'].append(filename)
            data['slice_no'].append(float(filename.split('_')[-1][:-4]))
            data['Prediction'].append(pred)
            data['Target'].append(target)
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Filename', 'slice_no']).reset_index(drop=True)
    csv_name = os.path.join(log_dir, 'Wrong_label.csv')
    df.to_csv(csv_name, index=False)


    # Save wrongly predicted images
    gts = [0,1,2]
    lbs = ['Non', 'Venous', 'Aterial']
    save_dir = os.path.join(log_dir, 'wrong_labeled')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for gt, lb in zip(gts, lbs):
        folder = os.path.join(save_dir, lb)
        if not os.path.isdir(folder):
            os.mkdir(folder)
            
        df_by_study = df[df.Target == gt]
        files = df_by_study.Filename.values
        tars = df_by_study.Target.values
        preds = df_by_study.Prediction.values

        for file, tar, pred in zip(files, tars, preds):
            study_id = file.split('/')[-2]
            
            study_folder = os.path.join(folder, study_id)
            if not os.path.isdir(study_folder):
                os.mkdir(study_folder)
            
            filename = file.split('/')[-1].split('.')[0]
            pred = lbs[pred]
            new_name = f'{filename}_P-{pred}.jpg'
            
            copyfile(file, os.path.join(study_folder, new_name))
        

    # Plotting confusion matrix 
    cm = confusion_matrix(y_target=targets, 
                      y_predicted=predictions, 
                      binary=False)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    fig_name = os.path.join(log_dir, f'{mode}_confusion_matrix_img_level.jpg')
    fig.savefig(fig_name)


