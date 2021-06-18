# -*- coding: utf-8 -*-

import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url

import models
import datasets

import numpy as np
import pandas as pd

from rdkit.Chem import AllChem as Chem

BASE_MODEL_DIR = Path("releases") / "download"
LATEST_VERSION = "v0.2-weights"

model_names = list(map(lambda x: x.name.split("_")[0], list((BASE_MODEL_DIR / LATEST_VERSION).iterdir())))
model_paths = dict(zip(map(lambda x: x.name.split("_")[0], list((BASE_MODEL_DIR / LATEST_VERSION).iterdir())),
                       map(lambda x: x.name, list((BASE_MODEL_DIR / LATEST_VERSION).iterdir()))))

def create_model(model_name, version=LATEST_VERSION, model_path=None, pretrained=False, pl_model=True, cache=True):

    if not model_name in model_names:
        raise ValueError("Expected one of model names({}), but got".format(",".join(model_names)))

    model_signature = getattr(models, model_name)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(f"{model_name}.yml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    model = model_signature(**config["model_kwargs"])
    state_getter = torch.load

    if pretrained:
        base_model_path = "."

        if not cache:
           base_model_path = "https://github.com/kukuleta/promol/tree/main/"
           state_getter = load_state_dict_from_url

        pretrained_model_path = '{base_path}/releases/download/{version}/{model_name}'.format(base_path=base_model_path,
                                                                                           version=version,
                                                                                           model_name=model_paths[model_name])

        state_dict = state_getter(pretrained_model_path, map_location=dev)

        if pl_model:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)

    model.to(dev)

    return model

def create_dataset_iterable(model_name, dataset_path, indexes=None, dataset_kwargs=None, loader_kwargs=None):

    with open(f"{model_name}.yml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    df = pd.read_csv(dataset_path).loc[:, ["SMILES", "Target Sequence", "Label"]].rename(columns={"SMILES": "smiles",
                                                                                                            "Target Sequence": "sequences",
                                                                                                             "Label": "affinities"})
    
    if not dataset_kwargs:
        dataset_kwargs = config["dataset_kwargs"]
    
    if not loader_kwargs:
        loader_kwargs = config["val_loader_kwargs"]

    if not indexes:
        indexes = df.index

    if model_name == "DeepDTA":
        df["smiles"] = df["smiles"].str.replace("\/", "").str.replace("@", "").str.replace("\\", "")

    if model_name == "DeepConvDTI":

        def get_morgan_fingerprint(x):
          try:
            rep = Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), radius=2, nBits=2048)

          except:
            return None
          
          else:
            return rep
            
        df.smiles = df.smiles.apply(get_morgan_fingerprint)
        df = df.drop(index=df.index[df["smiles"].isna()])
    
    dataset_signature = f"{model_name}Dataset"

    dataset = getattr(datasets, dataset_signature)(**{**{key: val.values for key, val in df.loc[indexes, 
                                                                                                        ["sequences", "smiles", "affinities"]]\
                                                                  .to_dict(orient="series").items()},
                                                                  **dataset_kwargs})
    dataset = DataLoader(dataset, **loader_kwargs)

    return dataset
