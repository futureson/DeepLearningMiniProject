# -*- coding: utf-8 -*-
import os
import json
import time
import random

from typing import Any, Dict

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

metrics = ["l1", "l2", "psnr_noised", "psnr_denoised"]

class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)

def create_montage(img_idx, save_path, source_t, denoised_t, clean_t, show=0):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    # fig.canvas.set_window_title(img_name.capitalize()[:-4])

    # Bring tensors to CPU
    source_t = source_t.cpu().narrow(0, 0, 3)
    denoised_t = denoised_t.cpu()
    clean_t = clean_t.cpu()
    
    source = tvF.to_pil_image(source_t)
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    clean = tvF.to_pil_image(clean_t)

    source_t = torch.unsqueeze(source_t, dim=0)
    denoised_t = torch.unsqueeze(denoised_t, dim=0)
    clean_t = torch.unsqueeze(clean_t, dim=0)

    # Build image montage
    psnr_vals = [psnr_noised(target=clean_t, source=source_t), psnr_denoised(target=clean_t, denoised=denoised_t)]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Ground truth']
    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img)
        ax[j].set_title(title)
        ax[j].axis('off')

    # Open pop up window, if requested
    if show > 0:
        plt.show()

    # Save to files
    # fname = os.path.splitext(img_name)[0]
    source.save(os.path.join(save_path, f'noisy-{img_idx}.png'))
    denoised.save(os.path.join(save_path, f'denoised-{img_idx}.png'))
    fig.savefig(os.path.join(save_path, f'montage-{img_idx}.png'), bbox_inches='tight')

def flip(input, target):
    """
    Implement the same horizontal and vertical flip operation onto the input and target.
    """
    if random.random() > 0.5:
        input = tvF.hflip(input)
        target = tvF.hflip(target)
    if random.random() > 0.5:
        input = tvF.vflip(input)
        target = tvF.vflip(target)
    return input, target

def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_batch_with_labels(imgs, label):
    if label == 0:
        input = imgs[0]
        target = imgs[1]
    elif label == 1:
        input = tensor_rot_90(imgs[0])
        target = tensor_rot_90(imgs[1])
    elif label == 2:
        input = tensor_rot_180(imgs[0])
        target = tensor_rot_180(imgs[1])
    elif label == 3:
        input = tensor_rot_270(imgs[0])
        target = tensor_rot_270(imgs[1])
    return input, target

def rotate(input, target):
	
	label = torch.randint(4, (1,), dtype=torch.long)
	return rotate_batch_with_labels((input, target), label)


class Logger(object):
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, folder_path: str) -> None:
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.folder_path = folder_path
        self.json_file_path = os.path.join(folder_path, "log-1.json")
        self.txt_file_path = os.path.join(folder_path, "log.txt")
        self.values = []

    def log_metric(
        self,
        name: str,
        values: Dict[str, Any],
        tags: Dict[str, Any],
        display: bool = False,
    ) -> None:
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})

        if display:
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def log(self, value: str, display: bool = True) -> None:
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        if display:
            print(content)
        self.save_txt(content)

    def save_json(self) -> None:
        """Save the internal memory to a file."""
        with open(self.json_file_path, "w") as fp:
            json.dump(self.values, fp, indent=" ")

        if len(self.values) > 1e4:
            # reset 'values' and redirect the json file to a different path.
            self.values = []
            self.redirect_new_json()

    def save_txt(self, value: str) -> None:
        with open(self.txt_file_path, "a") as f:
            f.write(value + "\n")

    def redirect_new_json(self) -> None:
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file for file in os.listdir(self.folder_path) if "json" in file
        ]
        self.json_file_path = os.path.join(
            self.folder_path, "log-{}.json".format(len(existing_json_files) + 1)
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min


class RuntimeTracker(object):
    """Tracking the runtime stat for local training."""

    def __init__(self, metrics_to_track):
        self.metrics_to_track = metrics_to_track
        self.reset()

    def reset(self):
        self.stat = dict((name, AverageMeter()) for name in self.metrics_to_track)

    def get_metrics_performance(self):
        return [self.stat[metric].avg for metric in self.metrics_to_track]

    def update_metrics(self, metric_stat, n_samples):
        for name, value in metric_stat.items():
            self.stat[name].update(value, n_samples)

    def __call__(self):
        return dict((name, val.avg) for name, val in self.stat.items())


class Metrics(object):
    def __init__(self, metrics) -> None:
        self._metrics = metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        self.tracker = RuntimeTracker(metrics_to_track=self._metrics)
        self._primary_metrics = self._metrics[1]  # psnr

    @torch.no_grad()
    def eval(self, source: torch.Tensor, denoised: torch.Tensor, target: torch.Tensor) -> None:
        results = dict()
        for metric_name in self._metrics:
            results[metric_name] = eval(metric_name)(target, denoised, source)
        self.tracker.update_metrics(results, n_samples=target.size(0))
        return results

l1_loss = nn.SmoothL1Loss()
l2_loss = nn.MSELoss()

def psnr_denoised(target, denoised=None, source=None, max_range=1.0):
    """Compute peak signal-to-noise ratio."""
    assert target.shape == denoised.shape and denoised.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((denoised-target) ** 2).mean((1,2,3))).mean()

def psnr_noised(target, denoised=None, source=None, max_range=1.0):
    """Compute peak signal-to-noise ratio."""
    assert target.shape == source.shape and source.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((source-target) ** 2).mean((1,2,3))).mean()

def l1(target, denoised=None, source=None):
    return l1_loss(denoised, target)

def l2(target, denoised=None, source=None):
    return l2_loss(denoised, target)

def lr_scheduler(optimizer, iter_ratio, gamma=10, power=0.75):
    decay = (1 + gamma * iter_ratio) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer

