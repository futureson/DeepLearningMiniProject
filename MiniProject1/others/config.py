# -*- coding: utf-8 -*-

config = {
    # general.
    "seed": 1,
    "log_dir": "./logs",
    # data.
    # "data_path": "./../../../data",
    # model.
    "resume": True,
    # Training.
    "loss": "l2",  # select among ["l1", "l2"].
    "lr": 0.0005,
    "optimizer": "Adam",
    "betas": [0.9, 0.99],
    "eps": 1e-8,
    "batch_size": 4,
    "epoch": 1,
}