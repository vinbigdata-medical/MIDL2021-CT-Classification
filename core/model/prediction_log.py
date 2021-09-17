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
import numpy as np


def record_output(
    cfg,
    mode: str,
    log_dir: str,
    study_IDs,
    seriesNumbers,
    targets: list,
    predictions: list,
    filenames: list,
):
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

    # Save all file prediction
    data = {
        "Study_ID": [],
        "Filename": [],
        "SeriesNumber": [],
        "Prediction": [],
        "Label": [],
    }

    data["Study_ID"] = study_IDs
    data["Filename"] = filenames
    data["SeriesNumber"] = seriesNumbers
    data["Prediction"] = predictions
    data["Label"] = targets
    df = pd.DataFrame(data)
    df.to_csv(f"study_evaluation/Prediction_of_{cfg.MODEL.NAME}.csv", index=False)

    # Record Correct prediction and False prediction into 2 csv files
    df_correct_prediction = df.loc[(df.Prediction == df.Label)].reset_index(drop=True)
    df_false_prediction = df.loc[~(df.Prediction == df.Label)].reset_index(drop=True)

    false_pred_csv_name = os.path.join(log_dir, "Wrong_label.csv")
    df_false_prediction.to_csv(false_pred_csv_name, index=False)

    correct_pred_csv_name = os.path.join(log_dir, "Correct_label.csv")
    df_correct_prediction.to_csv(correct_pred_csv_name, index=False)

    # Save wrongly predicted images
    gts = [0, 1, 2, 3]
    lbs = ["Non", "Venous", "Aterial", "Others"]
    save_dir = os.path.join(log_dir, "wrong_labeled")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for gt, lb in zip(gts, lbs):
        folder = os.path.join(save_dir, lb)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        df_by_study = df_false_prediction[df_false_prediction["Label"] == gt]
        files = df_by_study["Filename"].values
        tars = df_by_study["Label"].values
        preds = df_by_study["Prediction"].values
        seriesNumbers = df_by_study["SeriesNumber"].values

        for file, tar, pred, seriesNo in zip(files, tars, preds, seriesNumbers):
            study_id = file.split("/")[-2]

            study_folder = os.path.join(folder, study_id)
            if not os.path.isdir(study_folder):
                os.mkdir(study_folder)
            series_level_folder = os.path.join(study_folder, str(seriesNo))
            if not os.path.isdir(series_level_folder):
                os.mkdir(series_level_folder)

            filename = file.split("/")[-1].split(".")[0]
            pred = lbs[pred]
            new_name = f"{filename}_P-{pred}.jpg"

            copyfile(file, os.path.join(series_level_folder, new_name))

    # Plotting confusion matrix
    # lbs = ['Non', 'Venous', 'Arterial', 'Others']
    # print(np.unique(predictions))
    # cm = confusion_matrix(y_target=targets,
    #                   y_predicted=predictions,
    #                   binary=False)
    # fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True,
    #                   class_names=lbs,)
    # fig_name = os.path.join(log_dir, f'{mode}_confusion_matrix_img_level.jpg')
    # fig.savefig(fig_name)

    # fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True,
    #                   class_names=lbs, show_normed=True)
    # fig_name = os.path.join(log_dir, f'{mode}_confusion_matrix_img_level_normalized.jpg')
    # fig.savefig(fig_name)
