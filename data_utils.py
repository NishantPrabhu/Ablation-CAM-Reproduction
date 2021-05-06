
""" 
Dataloaders and such.

Only working with ImageNet for now.
"""

import os
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset


class ImagenetteDataset(Dataset):
    """ 
    Using fastai's Imagenette dataset to 
    generate localization predictions
    """

    def __init__(self, root, network_transform, normal_transform):
        self.data, self.labels = [], []
        self.network_transform = network_transform
        self.normal_transform = normal_transform
        folders = os.listdir(root)
        for f in folders:
            image_paths = [os.path.join(root, f, img) for img in os.listdir(os.path.join(root, f))]
            self.data.extend([Image.open(path).convert("RGB") for path in image_paths])
            self.labels.extend([int(f.split("_")[0]) for _ in range(len(image_paths))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        network_img = self.network_transform(self.data[idx])
        normal_img = self.normal_transform(self.data[idx])
        return network_img, normal_img, self.labels[idx]


def get_imagenet_dataloader(root, batch_size, network_transform, normal_transform):
    network_dset = ImagenetteDataset(root, network_transform, normal_transform)
    network_loader = DataLoader(network_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    return network_loader

def get_cifar10_dataloader(root, batch_size, network_transform, normal_transform):
    network_dset = datasets.CIFAR10(root="./data/cifar10", train=True, transform=network_transform, download=True)
    network_val_dset = datasets.CIFAR10(root="./data/cifar10", train=False, transform=network_transform, download=True)
    normal_dset = datasets.CIFAR10(root="./data/cifar10", train=True, transform=normal_transform, download=True)
    network_loader = DataLoader(network_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    network_val_loader = DataLoader(network_val_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    normal_loader = DataLoader(normal_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    return network_loader, network_val_loader, normal_loader