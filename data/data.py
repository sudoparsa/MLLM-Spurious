import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
from torchvision.datasets.folder import default_loader



# Create a custom dataset wrapper to include file paths
class ImageNetWithPaths(datasets.ImageNet):
    def __getitem__(self, index):
        # Get the original image and label
        original_tuple = super().__getitem__(index)
        # Get the image path
        path = self.samples[index][0]
        # Return image, label, and path
        return original_tuple + (path,)

class SpuriousDataset(Dataset):
    def __init__(self, path, transform=None):
        imgs = []
        targets = []

        subdirs = next(os.walk(path))[1]
        for subdir in subdirs:
            subdir_class = int(subdir.split('_')[2])
            for file in os.listdir(os.path.join(path, subdir)):
                imgs.append(os.path.join(path, subdir, file))
                targets.append(subdir_class)

        self.internal_idcs = torch.argsort(torch.LongTensor(targets))
        print(f'SpuriousImageNet - {len(subdirs)} classes - {len(imgs)} images')
        self.transform = transform
        self.imgs = imgs
        self.targets = targets
        self.included_classes = list(set(list(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        internal_idx = self.internal_idcs[idx].item()
        img = default_loader((self.imgs[internal_idx]))
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[internal_idx]
        return img, label

def get_selected_classes():
    return ['seat belt', 'balance beam', 'pole', 'cowboy hat', 'bathtub',
            'lighter','obelisk','gymnastic horizontal bar','grey whale','baby pacifier',
            'parallel bars','beer bottle','bikini','snorkel', 'saxophone',
            ]
