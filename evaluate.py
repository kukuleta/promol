# -*- coding: utf-8 -*-
"""test_and_evaluate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pGIm08cFUYjRXBFZ59vknBn34LpWY47r
"""

import os
import sys
import yaml
from pathlib import Path

from factory import *
from test_model import get_predictions

import torch
import random
from torch import nn

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_correlations(df):
    return df.corr()
    
def get_model_evaluation_metrics(predictions, actuals, threshold=0.5):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
    precision = true_positive_rate / (true_positive_rate + false_positive_rate)

    f1_score = 2 * precision * true_positive_rate / (true_positive_rate + precision + 0.00001)

    optimal_threshold = thresholds[5:][np.argmax(f1_score[5:])]

    threshold_preds = (predictions > threshold) * 1
    optimal_threshold_preds = (predictions > optimal_threshold) * 1

    auroc, auprc = metrics.auc(false_positive_rate, true_positive_rate), average_precision_score(actuals, predictions)
    confusion_matrix_optimal, confusion_matrix_thresholded = confusion_matrix(actuals, optimal_threshold_preds), confusion_matrix(actuals, threshold_preds)
    recall_optimal, recall_thresholded = recall_score(actuals, optimal_threshold_preds), recall_score(actuals, threshold_preds)
    precision_optimal, precision_thresholded = precision_score(actuals, optimal_threshold_preds), precision_score(actuals, threshold_preds)


    total_cumsum_optimal, total_cumsum_thresholded = sum(sum(confusion_matrix_optimal)), sum(sum(confusion_matrix_thresholded)) 

    accuracy_optimal = (confusion_matrix_optimal[0, 0] + confusion_matrix_optimal[1, 1]) / total_cumsum_optimal
    sensitivity_optimal = confusion_matrix_optimal[0, 0] / (confusion_matrix_optimal[0, 0] + confusion_matrix_optimal[0, 1])
    specificity_optimal = confusion_matrix_optimal[1, 1] / (confusion_matrix_optimal[1, 0] + confusion_matrix_optimal[1, 1])


    accuracy_thresholded = (confusion_matrix_thresholded[0, 0] + confusion_matrix_thresholded[1, 1]) / total_cumsum_thresholded
    sensitivity_thresholded = confusion_matrix_thresholded[0, 0] / (confusion_matrix_thresholded[0, 0] + confusion_matrix_thresholded[0, 1])
    specificity_thresholded = confusion_matrix_thresholded[1, 1] / (confusion_matrix_thresholded[1, 0] + confusion_matrix_thresholded[1, 1])

    res = {f"{str(threshold)}_threshold": {"accuracy": accuracy_thresholded,
                                           "sensivity": sensitivity_thresholded,
                                           "specifity": specificity_thresholded,
                                           "recall": recall_thresholded,
                                           "precision": precision_thresholded
                                           },
           "optimal_threshold": {"accuracy": accuracy_optimal,
                                 "sensivity": sensitivity_optimal,
                                 "specifity": specificity_optimal,
                                 "recall": recall_optimal,
                                 "precision": precision_optimal
                                 },
           "threshold": optimal_threshold,
           "auroc": auroc,
           "auprc": auprc}

    return res