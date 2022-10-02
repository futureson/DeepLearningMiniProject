# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import sys
sys.path.append('..')

import torch

from utils import dict2obj, Logger
from model import Model
from config import config


# config = {
#     # general.
#     "seed": 1,
#     "log_dir": "./logs",
#     "device": "cuda:0",  # e.g., cuda:0 if having gpu resources.
#     # data.
#     "data_path": "./../../../data",
#     # model.
#     "resume": True,
#     # Training.
#     "loss": "l2",  # select among ["l1", "l2"].
#     # "lr": 0.001,
#     "lr": 0.0005,
#     "optimizer": "AdamW",
#     "betas": [0.9, 0.99],
#     "eps": 1e-8,
#     "batch_size": 4,
#     "epoch": 2,
# }

def main(config):

    # set up logger.
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    log_path = os.path.join(config.log_dir, config.loss, TIME_NOW)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = Logger(folder_path=log_path)
    
    # load data.
    train_data_path = os.path.join(config.data_path, "train_data.pkl")
    val_data_path = os.path.join(config.data_path, "val_data.pkl")
    train_input, train_target = torch.load(train_data_path)
    val_input, val_target = torch.load(val_data_path)

    # Create model and training.
    logger.log("Constructing training model.")
    n2n = Model(config)

    # Training and evaluation.
    logger.log("Training begins.")
    n2n.train(train_input=train_input,
              train_target=train_target,
              num_epochs=config.epoch)
    n2n.eval(val_input=val_input,
             val_target=val_target)

    logger.save_json()

if __name__ == "__main__":
    main(dict2obj(config))
