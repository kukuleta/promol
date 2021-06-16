# -*- coding: utf-8 -*-

import os
import sys
import yaml
import copy
import inspect
import argparse
from time import time
from pathlib import Path

import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import classification
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import models
import datasets

parser = argparse.ArgumentParser(description='Training control parameters')
parser.add_argument('--seed', type=int, #nargs='+',
                    help='Seed for random function generator for reproducible experiments')
parser.add_argument('--dataset', type=str, #nargs='+',
                    help='Dataset path refering to fold indices and base dataset')
parser.add_argument('--model_name', type=str, #nargs='+',
                    help='Model type to train DTI pairs')
parser.add_argument('--model_save_dir', type=str, #nargs='+',
                    help='Path to save checkpoints')
parser.add_argument('--config_path', type=str, #nargs='+',
                    help='Path to .yml file specifying model training experiment configurations.')
parser.add_argument('--train_configuration', type=int, nargs=argparse.REMAINDER,
                    help='Training controller parameters such as epoch, learning_rate, etc.')

args = vars(parser.parse_args())
seed, dataset, model_name = args["seed"], args["dataset"], args["model_name"]
model_save_dir, config_path, train_parameters = args["model_save_dir"], args["config_path"], args["train_configuration"]

if not config_path:
  config_path = f"{model_name}.yml"

if not model_save_dir:
  config_path = f"models"

np.random.seed(seed)
torch.manual_seed(seed)

metric_list = ["roc_auc_score", "average_precision_score"]


def run_training_with_cv(X: pd.DataFrame,
                         model_signature,
                         model_kwargs,
                         dataset_signature,
                         dataset_kwargs,
                         train_loader_kwargs,
                         val_loader_kwargs,
                         trainer_kwargs,
                         seed=42,
                         cv_signature = None,
                         cv_kwargs = None,
                         checkpoint_callback_kwargs = None,
                         checkpoint_path:str = "models",
                         train_indices=None,
                         val_indices=None,
                         ):
  
    MODEL_CHECKPOINT_FOLDER = Path(checkpoint_path)
    SEED_FOLDER = MODEL_CHECKPOINT_FOLDER / f"SEED{seed}"
    CHECKPOINT_NAME_FORMAT = "{epoch}-{train_loss_epoch:.6f}-{val_loss_epoch:.6f}"

    MODEL_CHECKPOINT_FOLDER.mkdir(exist_ok=True)
    SEED_FOLDER.mkdir(exist_ok=True)

    if not checkpoint_callback_kwargs:
      checkpoint_callback_kwargs = {}
    
    if cv_signature:
      cv = cv_signature(**cv_kwargs)
      folds = cv.split(X, X["Label"])

    else:
      if train_indices and val_indices:
        folds = [(train_indices, val_indices)]

    

    for fold_id, (train_indices, validation_indices) in enumerate(folds):
      

        train_dataset, validation_dataset = [dataset_signature(**{**{key: val.values for key, val in X.loc[idx, 
                                                                                                        ["sequences", "smiles", "affinities"]]\
                                                                  .to_dict(orient="series").items()},
                                                                  **dataset_kwargs})
                                            for idx in [train_indices, validation_indices]]
                                            
        train_loader = DataLoader(train_dataset, 
                                  **train_loader_kwargs)
          

        val_loader = DataLoader(validation_dataset, 
                                **val_loader_kwargs)
      

        checkpoint_callback =  ModelCheckpoint(
                                              filename=os.path.join(str(SEED_FOLDER / f"fold{fold_id}"), "{epoch}-{train_loss_epoch:.6f}-{val_loss_epoch:.6f}-{roc_auc_score:.6f}"),
                                              save_top_k=3,
                                              save_weights_only=False,
                                              #save_last=True,
                                              verbose=False,
                                              monitor='roc_auc_score',
                                              mode='max')
        

        callbacks = [
                     EarlyStopping(
                         monitor='roc_auc_score',
                         min_delta=1e-6,
                         patience=11,
                         verbose=True,
                         mode='max',
                        strict=True),
                     
                     LearningRateMonitor(
                         logging_interval='step')
                     ]

        logger = TensorBoardLogger(SEED_FOLDER / f"fold{fold_id}",
                                   name="logs",
                                   default_hp_metric=False
                                   )
        
        
        
        trainer = Trainer(
                          logger=logger,
                          callbacks=callbacks + [checkpoint_callback],
                          **trainer_kwargs
                          )
                
        model = model_signature(**model_kwargs)


        trainer.fit(model, train_loader, val_loader)


with open(config_path, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

config["model_kwargs"].update({"evaluation_callbacks": metric_list,
                               "evaluation_thresholds": [0.5]})

df = pd.read_csv(dataset).loc[:, ["SMILES", "Target Sequence", "Label"]].rename(columns={"SMILES": "smiles",
                  "Target Sequence": "sequences",
                   "Label": "affinities"})

if model_name == "DeepDTA":
  df["smiles"] = df["smiles"].str.replace("\/", "").str.replace("@", "").str.replace("\\", "")

if config["predefined_folds"]:
  config["train_indices"] = np.fromfile(str(Path(dataset).parent.resolve() / "train_indices.bin"), dtype=int).tolist()
  config["val_indices"] = np.fromfile(str(Path(dataset).parent.resolve() / "val_indices.bin"), dtype=int).tolist()

config["dataset_signature"] = getattr(datasets, f"{model_name}Dataset")
config["model_signature"] = getattr(models, model_name)

function_keys = inspect.signature(run_training_with_cv).parameters.keys()
config={key:val for key, val in config.items() if key in function_keys}
config["trainer_kwargs"].update({"gpus": 0})
config["checkpoint_path"] = model_save_dir

run_training_with_cv(df,
                    **config)