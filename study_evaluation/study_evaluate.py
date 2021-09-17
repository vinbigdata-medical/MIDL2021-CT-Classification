import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sampling import sample
from statistics import mode
from collections import Counter
from multiprocessing import Pool

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# from series_loader import build_dataloader
from args import parse_args

# GLOBAL VARIABLES
n_bootstrap = 500
NUM_WORKERS = 8
target_names_dict = {"Non": 0, "Venous": 1, "Aterial": 2, "Others": 3}


def majority_voting(slice_result):
    c = Counter(slice_result)
    result = c.most_common(1)[0][0]
    return result


def classification_report(y_true, y_pred, target_names_dict):
    report = dict()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    v, c = np.unique(y_true, return_counts=True)
    count_sample = dict(zip(v, c))

    for name, value in target_names_dict.items():
        try:
            no_sample = count_sample[value]
        except KeyError:
            continue

        report_per_class = dict()

        bin_y_true = np.where(y_true != value, -1, y_true)
        bin_y_true = np.where(bin_y_true != -1, 1, bin_y_true)
        bin_y_true = np.where(bin_y_true == -1, 0, bin_y_true)

        bin_y_pred = np.where(y_pred != value, -1, y_pred)
        bin_y_pred = np.where(bin_y_pred != -1, 1, bin_y_pred)
        bin_y_pred = np.where(bin_y_pred == -1, 0, bin_y_pred)

        report_per_class["Precision"] = precision_score(bin_y_true, bin_y_pred)
        report_per_class["Recall"] = recall_score(bin_y_true, bin_y_pred)
        report_per_class["F1-Score"] = f1_score(bin_y_true, bin_y_pred)
        report_per_class["Support"] = no_sample

        report[name] = report_per_class
    return report


class evaluator:
    def __init__(self, test_csv: str):
        """evaluator object performs bootstrapping and
        scan level evaluation based on a slice-level prediction
        .csv file
        Args:
            test_csv (str): Slice level prediction file path
        """
        self.df = pd.read_csv(test_csv)
        self.unique_scan_df = (
            self.df[["Study_ID", "SeriesNumber", "Label"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.studyIDs = np.unique(self.df.Study_ID.values)
        self.lbs = [0, 1, 2]

        # self.trials = [1,5] + list(range(10, 40, 5)) + list(range(40,110,10)) # 5, 10, 15, 20 % of the total number of slices per scan
        self.chosen_ratio = 20
        self.trials = [
            self.chosen_ratio
        ]  # 5, 10, 15, 20 % of the total number of slices per scan

        self.data = dict()
        for chosen_ratio in self.trials:
            self.data[chosen_ratio] = []

    def evaluate_per_trial(self, seed):
        """[summary]
        Args:
            seed ([int]): seed value for the result to be
                        reproducible
        Returns:
            f1_scores [list]: list of f1 scores of 1 bootstrap sample
        """
        # np.random.seed(seed)

        data = dict()
        f1_scores = dict()
        reports = dict()
        for chosen_ratio in self.trials:
            f1_scores[chosen_ratio] = []
            reports[chosen_ratio] = []
            data[chosen_ratio] = ([], [])  # list of (predictions, labels)

        for i, row in self.unique_scan_df.iterrows():
            studyID = row.Study_ID
            lb = row.Label
            seriesNumber = row.SeriesNumber

            scan_df = self.df.loc[
                (self.df.Study_ID == studyID) & (self.df.SeriesNumber == seriesNumber)
            ].reset_index(drop=True)
            no_slices = len(scan_df)
            no_chosen_slice = int(
                no_slices * chosen_ratio / 100
            )  # When using PERCENTAGE
            # if no_chosen_slice < 1:
            #     no_chosen_slice = 1
            for chosen_ratio in self.trials:
                no_chosen_slice = chosen_ratio  # When using Slice No
                if no_chosen_slice > no_slices:
                    no_chosen_slice = no_slices

                rng = np.random.default_rng(seed)
                idxes = rng.choice(no_slices, size=no_chosen_slice, replace=False)

                slice_predictions = scan_df.Prediction.values[idxes]
                scan_prediction = majority_voting(slice_predictions)

                data[chosen_ratio][0].append(scan_prediction)
                data[chosen_ratio][1].append(lb)

        # print(np.unique(data[chosen_ratio][1], return_counts=True))

        for chosen_ratio in self.trials:
            f1 = f1_score(data[chosen_ratio][0], data[chosen_ratio][1], average="macro")
            f1_scores[chosen_ratio].append(f1)

            report = classification_report(
                data[chosen_ratio][1],
                data[chosen_ratio][0],
                target_names_dict=target_names_dict,
            )
            reports[chosen_ratio].append(report)

        return f1_scores, reports

    def evaluate(self):
        """Evaluate scan-level performance over n_bootstrap
        number of bootstraps
        """
        all_reports = {}
        for chosen_ratio in self.trials:
            all_reports[chosen_ratio] = []

        with Pool(processes=NUM_WORKERS) as p:
            max_ = n_bootstrap
            with tqdm(total=max_) as pbar:
                for i, (f1_scores, reports) in enumerate(
                    p.imap_unordered(self.evaluate_per_trial, range(n_bootstrap))
                ):
                    for chosen_ratio in self.trials:
                        self.data[chosen_ratio] += f1_scores[chosen_ratio]
                        all_reports[chosen_ratio] += reports[chosen_ratio]

                    pbar.update()

        # # PRINT AVERAGED REPORT
        dict_class = {"Non": 4, "Aterial": 7, "Venous": 10, "Others": 13}
        x = {"Precision": [], "Recall": [], "F1-Score": [], "Support": []}
        dict_vals = {
            "Non": {"Precision": [], "Recall": [], "F1-Score": [], "Support": []},
            "Venous": {"Precision": [], "Recall": [], "F1-Score": [], "Support": []},
            "Aterial": {"Precision": [], "Recall": [], "F1-Score": [], "Support": []},
            "Others": {"Precision": [], "Recall": [], "F1-Score": [], "Support": []},
        }
        reports = all_reports[self.chosen_ratio]

        for report in reports:
            for class_, dict_val in dict_vals.items():
                try:
                    report_by_class = report[class_]
                except KeyError:
                    report_by_class = {
                        "Precision": 0,
                        "Recall": 0,
                        "F1-Score": 0,
                        "Support": 0,
                    }

                for i, metric in enumerate(dict_val):
                    value = float(report_by_class[metric])
                    dict_val[metric].append(value)

        for key, val in dict_class.items():
            metrics = []
            dict_val = dict_vals[key]
            print(f" ===== {key} =====")
            for i, metric in enumerate(dict_val):
                mean_metric = np.mean(dict_val[metric])
                mean_metric = f"{mean_metric:.4f}"
                print(f"{metric}: {mean_metric}")

        data = pd.DataFrame(self.data)
        data.to_csv("bootstrap.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    test_csv = args.prediction_file

    evaluator_ = evaluator(test_csv)
    evaluator_.evaluate()
