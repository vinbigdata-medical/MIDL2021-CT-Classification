from tqdm import tqdm
import pandas as pd
import os
import torch
import torch.nn as nn
from shutil import copyfile
import cv2
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix, classification_report

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image


def record_output_gradcam(
    cfg,
    mode: str,
    log_dir: str,
    targets: list,
    predictions: list,
    filenames: list,
    model,
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

    target_layer = model.bn2
    # target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=cfg.SYSTEM.GPU)

    # Record wrongly predicted sample as df
    # df = pd.read_csv('Wrong_label.csv')
    data = {
        "StudyID": [],
        "Filename": [],
        "slice_no": [],
        "SeriesNumber": [],
        "Prediction": [],
        "Target": [],
    }
    data_correct_pred = {
        "StudyID": [],
        "Filename": [],
        "slice_no": [],
        "SeriesNumber": [],
        "Prediction": [],
        "Target": [],
    }
    for i, (target, pred, filename) in enumerate(zip(targets, predictions, filenames)):
        if target != pred:
            data["StudyID"].append(filename.split("/")[-2])
            data["Filename"].append(filename)
            data["SeriesNumber"].append(filename.split("/")[-1].split("_")[0])
            data["slice_no"].append(i)
            data["Prediction"].append(pred)
            data["Target"].append(target)
        else:
            if len(data_correct_pred["Filename"]) < 1000:
                data_correct_pred["StudyID"].append(filename.split("/")[-2])
                data_correct_pred["Filename"].append(filename)
                data_correct_pred["SeriesNumber"].append(
                    filename.split("/")[-1].split("_")[0]
                )
                data_correct_pred["slice_no"].append(i)
                data_correct_pred["Prediction"].append(pred)
                data_correct_pred["Target"].append(target)

    df = pd.DataFrame(data)
    df = df.sort_values(by=["Filename", "slice_no"]).reset_index(drop=True)
    csv_name = os.path.join(log_dir, "Wrong_label.csv")
    df.to_csv(csv_name, index=False)

    df_correct = pd.DataFrame(data_correct_pred)
    df_correct = df_correct.sort_values(by=["Filename", "slice_no"]).reset_index(
        drop=True
    )
    csv_name = os.path.join(log_dir, "Correct_label.csv")
    df_correct.to_csv(csv_name, index=False)

    # Save correctly predicted images
    gts = [0, 1, 2, 3]
    lbs = ["Non", "Venous", "Aterial", "Others"]
    save_dir = os.path.join(log_dir, "correct_labeled_gradcam")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for gt, lb in zip(gts, lbs):
        folder = os.path.join(save_dir, lb)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        df_by_study = df_correct[df_correct.Target == gt]
        files = df_by_study.Filename.values
        tars = df_by_study.Target.values
        preds = df_by_study.Prediction.values
        seriesNumbers = df_by_study.SeriesNumber.values

        for file, tar, pred, seriesNo in tqdm(zip(files, tars, preds, seriesNumbers)):
            img = cv2.imread(file)
            img = cv2.resize(img, cfg.DATA.SIZE, interpolation=cv2.INTER_AREA)
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            input_tensor = torch.Tensor(img).cuda()

            target_category = None
            grayscale_cam = cam(
                input_tensor=input_tensor, target_category=target_category
            )
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = np.squeeze(np.array(input_tensor.cpu())).transpose(1, 2, 0) / 255
            grad_cam_img = show_cam_on_image(rgb_img, grayscale_cam)
            # grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB)

            study_id = file.split("/")[-2]

            study_folder = os.path.join(folder, study_id)
            if not os.path.isdir(study_folder):
                os.mkdir(study_folder)
            series_level_folder = os.path.join(study_folder, seriesNo)
            if not os.path.isdir(series_level_folder):
                os.mkdir(series_level_folder)

            filename = file.split("/")[-1].split(".")[0]
            pred = lbs[pred]
            new_name = f"{filename}_P-{pred}.jpg"

            cv2.imwrite(os.path.join(series_level_folder, new_name), grad_cam_img)

    # Save wrongly predicted images
    gts = [0, 1, 2, 3]
    lbs = ["Non", "Venous", "Aterial", "Others"]
    save_dir = os.path.join(log_dir, "wrong_labeled_gradcam")
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
        seriesNumbers = df_by_study.SeriesNumber.values

        for file, tar, pred, seriesNo in tqdm(zip(files, tars, preds, seriesNumbers)):
            img = cv2.imread(file)
            img = cv2.resize(img, cfg.DATA.SIZE, interpolation=cv2.INTER_AREA)
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            input_tensor = torch.Tensor(img).cuda()

            target_category = None
            grayscale_cam = cam(
                input_tensor=input_tensor, target_category=target_category
            )
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = np.squeeze(np.array(input_tensor.cpu())).transpose(1, 2, 0) / 255
            grad_cam_img = show_cam_on_image(rgb_img, grayscale_cam)
            # grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB)

            study_id = file.split("/")[-2]

            study_folder = os.path.join(folder, study_id)
            if not os.path.isdir(study_folder):
                os.mkdir(study_folder)
            series_level_folder = os.path.join(study_folder, seriesNo)
            if not os.path.isdir(series_level_folder):
                os.mkdir(series_level_folder)

            filename = file.split("/")[-1].split(".")[0]
            pred = lbs[pred]
            new_name = f"{filename}_P-{pred}.jpg"

            cv2.imwrite(os.path.join(series_level_folder, new_name), grad_cam_img)
