import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader, Subset

class Data(Dataset):
    def __init__(self, cfg, mode:str):
        """A Dataset object that load all data for running

        Args:
            cfg (CfgNode): Config object containing running configuration
            mode (str): Model running mode
        """

        if mode == 'train':
            data_csv = pd.read_csv(cfg.DATA.CSV.TRAIN)
        elif mode == 'valid':
            data_csv = pd.read_csv(cfg.DATA.CSV.VALID)
        elif mode == 'test':
            data_csv = pd.read_csv(cfg.DATA.CSV.TEST)

        self.imgs = data_csv['Image'].values
        self.labels = data_csv['Label'].values
        self.cfg = cfg

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_dir = self.cfg.DIRS.DATA
        filename = f'{data_dir}/{self.imgs[idx]}'
        img = np.load(filename)

        img = img.transpose(2,0,1)
        label = self.labels[idx]

        img = np.expand_dims(img, axis=0)
        return filename,0, 0, img, label

def build_dataloader_3d(cfg, mode='train'):
    '''
    Build dataloader for 3D image

    Args:
        mode: Dataset type (train/valid/test dataset)
    Returns:
        dataloader: Dataloader object 
    '''

    dataset = Data(cfg, mode)
    # DEBUG: Only take a subset of dataloader to run script
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset, 
                        np.random.choice(np.arange(len(dataset)), 64))
    
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE, \
                            pin_memory=False, shuffle=True, \
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader