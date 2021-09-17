import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score

import pprint


def classification_report_(y_true, y_pred, target_names_dict):
    """Report performance on Accuracy, Precision, Recall and F1-Score

    Args:
        y_true (list): list of target, each element in the list is gt-label for a sample
        y_pred (list): list of prediction, each element in the list is prediction for a sample
        target_names_dict (dict): Dictionary -
                                    Key = Label (string) corresponding to 'predicted value'
                                    Value = Label predicted value (integer)

    Returns:
        [dict]: Report per class. Format
        {'Class Name': { 'Accuracy': Value, 'Precision': Value,
                        'Recall': Value}, 'F1-Score': Value }}
    """

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

        report_per_class["Accuracy"] = accuracy_score(bin_y_true, bin_y_pred)
        report_per_class["Precision"] = precision_score(bin_y_true, bin_y_pred)
        report_per_class["Recall"] = recall_score(bin_y_true, bin_y_pred)
        report_per_class["F1-Score"] = f1_score(bin_y_true, bin_y_pred)
        report_per_class["Support"] = no_sample

        report[name] = report_per_class
    return report


def print_report(report):
    macro = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Support": [],
    }

    for label, metrics in report.items():
        for metric, value in metrics.items():
            if value == []:
                continue
            macro[metric].append(value)

    overall_performance = "Overall Performance:"
    for metric, value in macro.items():
        if metric == "Support":
            macro_value = np.asarray(value).sum()
        else:
            macro_value = np.asarray(value).mean()
        macro[metric] = macro_value
        macro_value_str = f"{macro_value:.4f}"
        overall_performance += f"{metric}-{macro_value_str}  "
        # overall_performance += f'{metric}-{macro_value}  '

    print("===================================")
    print(overall_performance)
    print("===================================")

    pprint.pprint(report)
