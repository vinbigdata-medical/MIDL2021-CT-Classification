import numpy as np
import pandas as pd
import os
import timm
from tqdm import tqdm
import statistics
import time
import argparse
from sampling import sample
from statistics import mode

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from series_loader import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="", help="Device to run model on (cpu/gpu)"
    )
    parser.add_argument("--load", type=str, default="", help="model weight path")
    parser.add_argument("--model", type=str, default="resnet18", help="model name")
    parser.add_argument("--mode", type=str, default="valid", help="valid/test")
    parser.add_argument(
        "--sampling",
        type=str,
        default="random",
        help="Sampling type (all/half/random/centroid)",
    )
    parser.add_argument(
        "--slide_num",
        type=str,
        default=50,
        help="Number slide sampled for testing per study",
    )
    args = parser.parse_args()

    return args


# GLOBAL VARIABLES
args = parse_args()
n_bootstrap = 1


def build_model(model_name):
    model = timm.create_model(model_name, pretrained=False)
    num_class = 3
    if "resnet" in model_name:
        model.fc = nn.Linear(512, num_class)
    elif "efficientnet" in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    return model


def load_model(model, path):
    if os.path.isfile(path):
        ckpt = torch.load(path, "cpu")
        model.load_state_dict(ckpt.pop("state_dict"))
        start_epoch, best_metric = ckpt["epoch"], ckpt["best_metric"]

    return model


def predict_series(model, df):
    """
    Args:
        study_series: Path to Series
    Return:
        pred: predicted class
    """
    loader = build_dataloader(df, args=args)
    preds = list()

    for image in loader:
        with torch.no_grad():
            # image = torch.tensor(image).float().cuda()
            image = image.float().cuda()
            output = model(image)
        sigmoid = nn.Sigmoid()
        pred = torch.argmax(sigmoid(output), 1)

        # Convert target, prediction to numpy
        pred = list(pred.detach().cpu().numpy())
        preds += pred

    series_pred = mode(preds)
    return series_pred


def evaluate(model, test_csv):
    model.eval()
    model.cuda()

    test = pd.read_csv(test_csv)
    filenames = test.Image.values
    study_ids = [name.split("/")[-2] for name in filenames]
    test["studyID"] = study_ids

    targets, preds = list(), list()

    lbs = [0, 1, 2]
    total_pred_time, num_series = 0, 0

    for study_series in tqdm(np.unique(study_ids)):
        # print(study_series)
        for lb in lbs:
            df = test[(test.studyID == study_series) & (test.Label == lb)].reset_index(
                drop=True
            )
            if len(df) == 0:
                continue

            # Predict series
            start = time.time()
            pred = predict_series(model, df)
            end = time.time()
            total_pred_time += end - start
            num_series += 1

            targets.append(lb)
            preds.append(pred)
    # Calculate Metrics
    accuracy = accuracy_score(targets, preds)
    recall = recall_score(targets, preds, average="macro")
    precision = precision_score(targets, preds, average="macro")
    f1 = f1_score(targets, preds, average="macro")
    print(
        "ACCURACY: %.9f, RECALL: %.9f, PRECISION: %.9f, F1: %.9f"
        % (accuracy, recall, precision, f1)
    )
    report = classification_report(
        targets, preds, target_names=["Non", "Aterial", "Venous"]
    )
    print(report)

    # PREPARE LOGGING DIRECTORY
    log_dir = "../outputs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.load.split("/")[-1][:-4])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Plotting confusion matrix
    cm = confusion_matrix(y_target=targets, y_predicted=preds, binary=False)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    fig_name = os.path.join(log_dir, "valid_confusion_matrix_series_level.jpg")
    fig.savefig(fig_name)

    # Prediction time
    print("Series Average prediction time:", total_pred_time / num_series)

    return accuracy, recall, precision


def slide_num_search():
    np.random.seed(42)
    trials = range(5, 50, 5)

    data = {}
    for key in trials:
        data[key] = []

    model_name = args.model
    ckp_path = args.load
    eval_mode = args.mode
    test_csv = f"../Split_data/data_delay/{eval_mode}.csv"

    model = build_model(model_name)
    model_ckp = load_model(model, ckp_path)

    for seed in tqdm(range(n_bootstrap)):
        np.random.seed(seed)
        for val in tqdm(trials):
            args.slide_num = val
            accuracy, recall, precision = evaluate(model_ckp, test_csv)
            data[val].append(accuracy)

    df = pd.DataFrame(data)
    sampling_mode = args.sampling
    df.to_csv(f"Eval_{eval_mode}.csv", index=False)


if __name__ == "__main__":
    model_name = args.model
    ckp_path = args.load
    eval_mode = args.mode
    test_csv = f"../Split_data/data_delay/{eval_mode}.csv"

    model = build_model(model_name)
    model_ckp = load_model(model, ckp_path)

    evaluate(model_ckp, test_csv)

    # slide_num_search()
