from tqdm import tqdm 
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score 
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from core.utils import AverageMeter
from core.utils import save_checkpoint
from .prediction_log import record_output

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

def valid_model(cfg, mode, epoch, model, 
                dataloader, criterion, writer, 
                save_prediction=False, best_metric=None):
    """Evaluate model performance on Validating dataset

    Args:
        cfg (CfgNode): Config object containing running configuration
        mode (str): Model running mode (valid/test)
        model (nn.Module): Model that need to have performance evaluated
        dataloader (data.DataLoader): Dataloader object to load data batch-wise
        criterion: Loss function
        writer (Summarywriter): Logger that log validation loss and plot it on Tensorboard
        save_prediction (Boolean): Whether to save prediction output or not (for bootstraping)
        best_metric (float, optional): Best performance result of loaded model. Defaults to None.
    """

    # Declare variables
    gpu = cfg.SYSTEM.GPU
    output_log_dir = cfg.DIRS.OUTPUTS
    model.eval()
    losses = AverageMeter()
    tbar = tqdm(dataloader)
    targets, preds, filenames, study_IDs, seriesNumbers = list(), list(), list(), list(), list()
    data = dict()
    
    for i, (filename, study_ID, seriesNumber, image, target) in enumerate(tbar):
        with torch.no_grad():
            image = image.float()
            if gpu:
                image, target = image.cuda(), target.cuda()
                output = model(image)

            # Compute loss
            loss = criterion(output, target)
            sigmoid = nn.Sigmoid()
            pred = torch.argmax(sigmoid(output), 1)

            # Record loss
            losses.update(loss.item() * cfg.SOLVER.GD_STEPS, target.size(0))
            tbar.set_description("Valid loss: %.9f" % (losses.avg))

            # Convert target, prediction to numpy
            target = list(target.detach().cpu().numpy())
            pred = list(pred.detach().cpu().numpy())
            filename = list(filename)
            targets += target
            preds += pred
            filenames += filename
            study_IDs += study_ID
            seriesNumbers += seriesNumber

    # Record wrongly predicted sample and save confusion matrix
    # record_output(cfg, mode, output_log_dir, targets, preds, filenames)

    # Calculate Metrics
    accuracy = accuracy_score(targets, preds)
    recall = recall_score(targets, preds, average='macro')
    precision = precision_score(targets, preds, average='macro')
    f1 = f1_score(targets, preds, average='macro')
    print("ACCURACY: %.9f, RECALL: %.9f, PRECISION: %.9f, F1: %.9f" 
        % (accuracy, recall, precision, f1))
    report = classification_report(targets, preds, \
                            target_names=['Non','Aterial','Venous'])
    print(report)

    if mode == 'train':
        writer.add_scalars(f'Metrics', {
                    'F1_SCORE': f1,
                    'ACCURACY': accuracy,
                    'RECALL': recall, 
                    'PRECISION': precision
                }, epoch)

        # CHECKPOINT
        is_best = accuracy > best_metric
        best_metric = max(accuracy, best_metric)

        save_dict = {"epoch": epoch + 1,
                    "arch": cfg.NAME,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric}
        save_filename = f'{cfg.NAME}.pth'
        if is_best:
            save_checkpoint(save_dict, root=cfg.DIRS.WEIGHTS,
                            filename=save_filename)

    if save_prediction:
        # Save All slices prediction for scan prediction and bootstraping
        data['Study_ID'] = study_IDs
        data['Filename'] = filenames
        data['SeriesNumber'] = seriesNumbers
        data['Prediction'] = preds
        data['Label'] = targets
        data = pd.DataFrame(data)
        data.to_csv(f'study_evaluation/eval_{mode}.csv', index=False)