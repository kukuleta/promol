# -*- coding: utf-8 -*-
"""data_collector.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NUf189--tfKgW0xosjwNKfZuMZpMIX3E
"""

import os
import sys
import urllib.request as request
import string
from pathlib import Path


import pandas as pd

"""#### Evaluation Benchmark dataset collection in MolTrans Paper"""

data_path = Path("benchmarks")
data_path.mkdir(exist_ok=True)

datasets = ["BIOSNAP", "BindingDB", "DAVIS"]
dataset_identifiers = ["train", "test", "val"]
file_type = "csv"

BASE_URL = "https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/"

Path("repurpose").mkdir(exist_ok=True)
request.urlretrieve("https://dataverse.harvard.edu/api/access/datafile/4159648", Path("repurpose")  / ("broad_institute_repurpose_hub.tsv"))
request.urlretrieve("https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt", Path("repurpose")  / ("clue_repurpose_drugs.txt"))
request.urlretrieve("https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20180907.txt", Path("repurpose")  / ("clue_repurpose_metadata.txt"))


for dataset in datasets:
    (data_path / dataset).mkdir(exist_ok=True)
    for data_identifier in dataset_identifiers:
        file_name = data_identifier + "." + file_type
        if dataset == "BIOSNAP":
            file_category = "full_data" 
            file_ids = [file_category, file_name]
            request.urlretrieve(BASE_URL + "/".join([dataset, file_category, file_ids[1]]), data_path / dataset / ("_".join(file_ids)))
            filename = "_".join(file_ids)
        
        else:
            request.urlretrieve(BASE_URL + "/".join([dataset, file_name]), data_path / dataset / (file_name))

    base_path = data_path / dataset
    filename = "{data_identifier}.csv"


    if dataset == "BIOSNAP":
      filename = "full_data" + "_" + filename
    
    train, val, test = [pd.read_csv(base_path / filename.format(data_identifier=data_identifier)) for data_identifier in dataset_identifiers]


    train["dataset_type"] = "train"
    val["dataset_type"] = "val"
    test["dataset_type"] = "test"

    all_dataset = pd.concat([train, val, test]).reset_index(drop=True)

    del train, val, test

    for dataset_type in dataset_identifiers:
        all_dataset[all_dataset["dataset_type"].isin([dataset_type])].index.values.tofile(data_path / dataset / (dataset_type + "_" + "indices.bin"))

        filename = dataset_type + ".csv"

        if dataset == "BIOSNAP": 
          filename = "full_data" + "_" + filename
        
        os.remove(data_path / dataset / filename)
            
    all_dataset.to_csv(data_path / dataset / "data.csv")

(data_path / "AllDB").mkdir(exist_ok=True)

pd.concat([pd.read_csv(f"benchmarks/{dataset}/data.csv",
                       usecols=["SMILES", "Target Sequence", "Label"]) \
           for dataset in ["BIOSNAP", "BindingDB", "DAVIS"]]).drop_duplicates().to_csv((data_path / "AllDB") / "data.csv")

from sklearn.model_selection import train_test_split

def train_val_test_split(X,
                         y=None,
                         test_size=None,
                         train_size=None,
                         validation_size=None,
                         random_state=None,
                         shuffle=True,
                         stratify=None
                         ):
  
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size=1 - train_size, 
                                                        random_state=random_state,
                                                        shuffle=shuffle, 
                                                        stratify=y)
    

    X_val, X_test, Y_val, Y_test = train_test_split(X_test,
                                                    Y_test,
                                                    test_size=test_size/(test_size + validation_size),
                                                    random_state=random_state,
                                                    shuffle=shuffle, 
                                                    stratify=Y_test) 
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

unified_data = pd.read_csv("/content/benchmarks/AllDB/data.csv", index_col=0)
X_train, X_val, X_test, Y_train, Y_val, Y_test = train_val_test_split(unified_data.drop(columns=["Label"]),
                     unified_data["Label"],
                     test_size=0.2,
                     validation_size=0.1,
                     train_size=0.7,
                     random_state=42)
X_train["Dataset"] = "train"
X_test["Dataset"] = "test"
X_val["Dataset"] = "val"

data = pd.concat([X_train, X_test, X_val])
data["Label"] = pd.concat([Y_train, Y_test, Y_val])
data = data.reset_index(drop=True)

data.query("Dataset == 'train'").index.values.tofile(data_path / "AllDB" / ("train" + "_" + "indices.bin"))
data.query("Dataset == 'test'").index.values.tofile(data_path / "AllDB" / ("test" + "_" + "indices.bin"))
data.query("Dataset == 'val'").index.values.tofile(data_path / "AllDB" / ("val" + "_" + "indices.bin"))

data.to_csv("/content/benchmarks/AllDB/data.csv", index=False)