from pathlib import Path
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

from .others.networks import UNet
from .others.utils import dict2obj, Logger, Metrics, lr_scheduler
from .others.datasets import define_training_set, define_validation_set
from .others.config import config

class Model():
    def __init__(self) -> None:
        self._meta_conf = dict2obj(config)
        # Set up device.
        self._meta_conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up seed.
        if not hasattr(self._meta_conf, "seed"):
            self._meta_conf.seed = 1
        torch.manual_seed(self._meta_conf.seed)
        # instantiate model.
        self.initialize_model()
        # initialize optimizer.
        self.optimizer = self.initialize_optimizer(self.model.parameters(), self._meta_conf)
        # add a lr scheduler.
        for param_group in self.optimizer.param_groups:
            param_group['lr0'] = param_group['lr']

        # instantiate loss function.
        self.criterion = self.initialize_criterion(self._meta_conf)

        # initialize metrics.
        self._metrics_to_track = [self._meta_conf.loss, "psnr_noised", "psnr_denoised"]
        self._metrics = Metrics(self._metrics_to_track)

        # set up logger.
        if not hasattr(self._meta_conf, "log_dir"):
            self._meta_conf.log_dir = "./logs"
        DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        TIME_NOW = datetime.now().strftime(DATE_FORMAT)
        log_path = os.path.join(self._meta_conf.log_dir, self._meta_conf.loss, TIME_NOW)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.logger = Logger(folder_path=log_path)
        self.debug=False  # If some results need to be printed out, set self.debug=True.

        torch.set_grad_enabled(True)

    def initialize_model(self):
        # Use U-Net as backbone network.
        model = UNet()
        self.model = model.to(self._meta_conf.device)

    def initialize_optimizer(self, params, meta_conf):
        """This function is used to initialize the optimizer."""
        if not hasattr(meta_conf, "optimizer"):
            meta_conf.optimizer = "Adam"
        if not hasattr(meta_conf, "lr"):
            meta_conf.lr = 0.0005
        if not hasattr(meta_conf, "betas"):
            meta_conf.betas = [0.9, 0.99]
        if not hasattr(meta_conf, "eps"):
            meta_conf.betas = 1e-8

        if meta_conf.optimizer == "Adam":
            return optim.Adam(
                params,
                lr=meta_conf.lr,
                betas=meta_conf.betas,
                eps=meta_conf.eps
            )
        elif meta_conf.optimizer == "AdamW":
            return optim.AdamW(
                params=params,
                lr=meta_conf.lr,
                betas=meta_conf.betas,
                eps=meta_conf.eps
            )
        elif meta_conf.optimizer == "SGD":
            return optim.SGD(
                params,
                lr=meta_conf.lr,
                momentum=meta_conf.momentum if hasattr(meta_conf, "momentum") else 0.9,
                dampening=meta_conf.dampening if hasattr(meta_conf, "dampening") else 0,
                weight_decay = meta_conf.weight_decay if hasattr(meta_conf, "weight_decay") else 0,
                nesterov=meta_conf.nesterov if hasattr(meta_conf, "nesterov") else True,
            )
        else:
            raise NotImplementedError

    def initialize_criterion(self, meta_conf):
        """This function is used to set up criterion according to meta_conf."""
        if not hasattr(meta_conf, "loss"):
            meta_conf.loss = "l2"

        if meta_conf.loss == "l2":
            return nn.MSELoss()
        elif meta_conf.loss == "l1":
            return nn.SmoothL1Loss()
        else:
            raise NotImplementedError

    def get_metrics_performance(self):
        """This is function is used to read metrics."""
        return self._metrics.tracker.get_metrics_performance()

    def reset_tracker(self):
        """This function is used to reset metrics after running evaluation on evaluation splits."""
        self._metrics.tracker.reset()

    def load_pretrained_model(self):
        """
        load the best pretrained model from checkpoint files.
        """
        ckpt_path = Path(__file__).parent / "bestmodel.pth"
        best_model = torch.load(ckpt_path, map_location=self._meta_conf.device)

        self.model.load_state_dict(best_model["state_dict"])

    def train(self, train_input, train_target, num_epochs):
        """Training denoiser."""
        if not hasattr(self._meta_conf, "batch_size"):
            self._meta_conf.batch_size = 4
        if not hasattr(self._meta_conf, "resume"):
            self._meta_conf.resume = True
        

        num_imgs = train_input.size()[0]
        num_batches = int(num_imgs / self._meta_conf.batch_size)

        # build dataloader.
        train_loader = define_training_set(train_input, train_target, self._meta_conf.device)

        # load state_dict of the best model. 
        if self._meta_conf.resume:
            self.load_pretrained_model()

        # begin training.
        self.model.train()
        for epoch in range(num_epochs):
            self.logger.log('\nEPOCH {:d} / {:d}'.format(epoch + 1, self._meta_conf.epoch), display=self.debug)
            for step, epoch_fractional, batch in train_loader.iterator(
            batch_size=self._meta_conf.batch_size,
            shuffle=True,
            repeat = False,
            ref_num_data=None,
            num_workers=self._meta_conf.num_workers
            if hasattr(self._meta_conf, "num_workers")
            else 1,
            pin_memory=True,
            drop_last=False,
        ):
                # Comment it if not using continuously decreasing lr strategy.
                lr_scheduler(self.optimizer, iter_ratio=(step + num_batches*epoch) / (self._meta_conf.epoch*num_batches))
                if not self._meta_conf.device == "cpu":
                    input = batch._x.to(self._meta_conf.device)
                    target = batch._y.to(self._meta_conf.device)
                else:
                    input = batch._x
                    target = batch._y

                # denoise image.
                input_denoised = self.model(input)
                loss_input = self.criterion(input_denoised, target)

                target_denoised = self.model(target)
                loss_target = self.criterion(input, target_denoised)

                loss = 0.5*loss_input + 0.5*loss_target

                # optimization.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval(self, val_input, val_target):
        """ evaluate the trained model on the evaluation split."""
        val_loader = define_validation_set(val_input, val_target, self._meta_conf.device)

        for step, epoch_fractional, batch in val_loader.iterator(
            batch_size=self._meta_conf.batch_size,
            shuffle=False,
            repeat = False,
            ref_num_data=None,
            num_workers=self._meta_conf.num_workers
            if hasattr(self._meta_conf, "num_workers")
            else 1,
            pin_memory=True,
            drop_last=False,
        ):
            if not self._meta_conf.device == "cpu":
                source = batch._x.to(self._meta_conf.device)
                target = batch._y.to(self._meta_conf.device)
            else:
                source = batch._x
                target = batch._y

            target = target.float() / 255.0

            denoised = self.predict(source)
            denoised = denoised / 255.0
            source = source.float() / 255.0
            self._metrics.eval(source, denoised, target)
        
        # retrive validation results.
        val_results = self.get_metrics_performance()
        self.logger.log(f'Validation loss: {val_results[0]}, Validation noised PSNR: {val_results[1]}, Validation denoised PSNR: {val_results[2]}', display=self.debug)
        self.reset_tracker()
        return val_results[0], val_results[2]

    def predict(self, test_input):
        """Testing using unseen images."""
        test_input = test_input.float() / 255.0
        test_input = test_input.to(self._meta_conf.device)

        y_hat = self.model(test_input)
        y_hat = torch.clamp(y_hat, 0, 1)
        return y_hat * 255.0

