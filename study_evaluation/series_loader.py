import numpy as np
import pandas as pd
import cv2
from sampling import sample


from torch.utils.data import Dataset, DataLoader, Subset

# For validation
class SeriesLoader(Dataset):
    def __init__(self, data_csv, args):
        self.imgs = data_csv["Image"].values
        self.labels = data_csv["Label"].values

        chosen_slices = sample(args, len(self.imgs))
        self.imgs = self.imgs[chosen_slices]
        self.labels = self.labels[chosen_slices]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_dir = "../Split_data"
        img_path = f"{data_dir}/{self.imgs[idx]}"
        img = cv2.imread(img_path)
        img = img.transpose(2, 0, 1)
        label = self.labels[idx]

        # Augmentation

        return img


def build_dataloader(data_csv, args, debug=False):
    """
    Build dataloader

    Returns:
        dataloader: Dataloader object
    """

    dataset = SeriesLoader(data_csv, args)
    # DEBUG: Only take a subset of dataloader to run script
    if debug:
        dataset = Subset(dataset, np.random.choice(np.arange(len(dataset)), 64))

    dataloader = DataLoader(
        dataset, 64, pin_memory=False, shuffle=True, drop_last=False, num_workers=8
    )
    return dataloader
