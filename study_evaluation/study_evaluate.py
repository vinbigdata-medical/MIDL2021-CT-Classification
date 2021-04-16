import numpy as np 
import pandas as pd
import os
from tqdm import tqdm
from sampling import sample
from statistics import mode
from multiprocessing import Pool

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score 
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from series_loader import build_dataloader
from args import parse_args

# GLOBAL VARIABLES
n_bootstrap = 1000
NUM_WORKERS = 8

class evaluator():
    def __init__(self, test_csv:str):
        """evaluator object performs bootstrapping and
        scan level evaluation based on a slice-level prediction
        .csv file 

        Args:
            test_csv (str): Slice level prediction file path
        """
        self.df = pd.read_csv(test_csv)
        self.unique_scan_df = self.df[['Study_ID', 'SeriesNumber', 'Label']].drop_duplicates().reset_index(drop=True)
        self.studyIDs = np.unique(self.df.Study_ID.values)
        self.lbs = [0,1,2]
        self.trials = list(range(40, 70, 10)) # 5, 10, 15, 20 % of the total number of slices per scan
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
        np.random.seed(seed)

        data = dict()
        f1_scores = dict()
        for chosen_ratio in self.trials:
            f1_scores[chosen_ratio] = []
            data[chosen_ratio] = ([], []) # list of (predictions, labels)

        for i, row in self.unique_scan_df.iterrows():
            studyID = row.Study_ID
            lb = row.Label
            seriesNumber = row.SeriesNumber

            scan_df = self.df.loc[(self.df.Study_ID == studyID) & (self.df.SeriesNumber == seriesNumber)].reset_index(drop=True)
            no_slices = len(scan_df)
            # no_chosen_slice = int(no_slices * chosen_ratio / 100) # When using PERCENTAGE
            # if no_chosen_slice < 1:
            #     no_chosen_slice = 1
            for chosen_ratio in self.trials:
                no_chosen_slice = chosen_ratio # When using Slice No
                if no_chosen_slice > no_slices:
                    no_chosen_slice = no_slices

                rng = np.random.default_rng()
                if no_slices < 10:
                    continue
                idxes = rng.choice(no_slices, size=no_chosen_slice, replace=False)

                slice_predictions = scan_df.Prediction.values[idxes]
                scan_prediction = mode(slice_predictions)

                data[chosen_ratio][0].append(scan_prediction)
                data[chosen_ratio][1].append(lb)

        for chosen_ratio in self.trials:
            f1 = f1_score(data[chosen_ratio][0], data[chosen_ratio][1],
                                 average='macro')
            f1_scores[chosen_ratio].append(f1)

        return f1_scores

    def evaluate(self):
        """Evaluate scan-level performance over n_bootstrap
        number of bootstraps
        """
        with Pool(processes=NUM_WORKERS) as p:
            max_ = n_bootstrap
            with tqdm(total=max_) as pbar:
                for i, f1_scores in enumerate(p.imap_unordered(self.evaluate_per_trial, range(n_bootstrap))):
                    for chosen_ratio in self.trials:
                        self.data[chosen_ratio] += f1_scores[chosen_ratio] 
                    pbar.update()

        data = pd.DataFrame(self.data)
        data.to_csv('bootstrap_168.csv', index=False)      
    
if __name__ == '__main__':
    args = parse_args()
    test_csv = args.prediction_file

    evaluator_ = evaluator(test_csv)
    evaluator_.evaluate()

    