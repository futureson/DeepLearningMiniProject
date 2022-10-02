# -*- coding: utf-8 -*-

from typing import Callable, Optional, Iterable, Tuple
import itertools
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
# import torchvision.transforms as transforms

from .utils import rotate, flip

class Batch(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return len(self._x)

    def to(self, device) -> "Batch":
        return Batch(self._x.to(device), self._y.to(device))

    def __getitem__(self, index):
        return self._x[index], self._y[index]


class TrainDataset(Dataset):
    """
    A base dataset class inherited from torch.utils.data.Dataset.
    """
    def __init__(self, noisy_imgs_1, noisy_imgs_2) -> None:
        # super().__init__()
        self.noisy_imgs_1 = noisy_imgs_1
        self.noisy_imgs_2 = noisy_imgs_2
        self.transform = [rotate]

    def __len__(self):
        return len(self.noisy_imgs_1)

    def __getitem__(self, index):
        input = self.noisy_imgs_1[index] / 255.0
        target = self.noisy_imgs_2[index] / 255.0
        num_aug = torch.randint(1, 2, (1,), dtype=int).item()
        idx_aug = torch.randperm(num_aug)
        for i in range(num_aug):
            input, target = self.transform[idx_aug[i]](input, target)

        return input, target


class ValDataset(Dataset):
    """
    A base dataset class inherited from torch.utils.data.Dataset.
    """
    def __init__(self, noisy_imgs_1, noisy_imgs_2) -> None:
        # super().__init__()
        self.noisy_imgs_1 = noisy_imgs_1
        self.noisy_imgs_2 = noisy_imgs_2

    def __len__(self):
        return len(self.noisy_imgs_1)

    def __getitem__(self, index):
        input = self.noisy_imgs_1[index]
        target = self.noisy_imgs_2[index]

        return input, target


class PyTorchDataset(object):
    def __init__(
        self,
        dataset,
        device,
        prepare_batch: Callable,
    ) -> None:
        self._set = dataset
        self._device = device
        self._prepare_batch = prepare_batch
        
    def __len__(self):
        return self._set

    @property
    def dataset(self):
        return self._set
    
    def iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        ref_num_data: Optional[int] = None,
        num_workers: int = 1,
        sampler: Optional[torch.utils.data.Sampler] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Iterable[Tuple[int, float, Batch]]:
        _num_batch = 1 if not drop_last else 0
        if ref_num_data is None:
            num_batches = int(len(self.dataset) / batch_size + _num_batch)
        else:
            num_batches = int(ref_num_data / batch_size + _num_batch)
        if sampler is not None:
            shuffle = False

        loader = DataLoader(
            self._set,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
        )

        step = 0
        for _ in itertools.count() if repeat else [0]:
            for i, batch in enumerate(loader):
                step += 1
                epoch_fractional = float(step) / num_batches
                yield step, epoch_fractional, self._prepare_batch(batch, self._device)


class NoisyDataset(PyTorchDataset):
    """
    A class to load the noisy training dataset.
    """

    def __init__(self, dataset, device) -> None:
        super().__init__(
            dataset=dataset, 
            device=device, 
            prepare_batch=NoisyDataset.prepare_batch
            )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class NormalLoader(object):
    """Define dataloader."""

    def __init__(self, dataset: NoisyDataset):
        self.dataset = dataset

    def iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        ref_num_data: Optional[int] = None,
        num_workers: int = 1,
        sampler: Optional[torch.utils.data.Sampler] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Iterable[Tuple[int, float, Batch]]:
        yield from self.dataset.iterator(
            batch_size,
            shuffle,
            repeat,
            ref_num_data,
            num_workers,
            sampler,
            pin_memory,
            drop_last,
        )

def define_training_set(input, target, device):
    """Instantiate datasets."""

    base_dataset = TrainDataset(input, target)
    dataset = NoisyDataset(base_dataset, device=device)
    dataset_loader = NormalLoader(dataset)
    return dataset_loader

def define_validation_set(input, target, device):
    """Instantiate datasets."""

    base_dataset = ValDataset(input, target)
    dataset = NoisyDataset(base_dataset, device=device)
    dataset_loader = NormalLoader(dataset)
    return dataset_loader

