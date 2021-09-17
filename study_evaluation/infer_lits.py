import numpy as np
import pandas as pd
import os
import timm
from tqdm import tqdm
from statistics import mode
import glob

import torch
import torch.nn as nn


from series_loader import build_dataloader
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import cv2


model_name = "efficientnet_b2"
ckp_path = "/home/single1/BACKUP/binhdao/binhdao/weights/EfficientNet_b2_224.pth"

label_dict = {0: "Non Contrast", 1: "Venous", 2: "Arterial", 3: "Others"}

# data = glob.glob('/home/single1/BACKUP/binhdao/lits/Training_Batch1/media/nas/01_Datasets/CT/LITS/Training_Batch_1/volume*')
data = glob.glob(
    "/home/single1/BACKUP/binhdao/lits/Training_Batch2/media/nas/01_Datasets/CT/LITS/Training_Batch_2/volume*"
)


window_width = 400
window_center = 50

def apply_window(img, ww: float, wc: float):
    """
    Apply HU window on a HU image

    Args:
        img: Image to transform
        ww: Window width
        wc: Window center
    """
    lower_bound = wc - ww / 2
    upper_bound = wc + ww / 2

    img[img < lower_bound] = lower_bound
    img[img > upper_bound] = upper_bound

    img = (img - wc) / ww * (upper_bound - lower_bound) + lower_bound

    return img


def preprocess_image(
    img,
    ww: float,
    wc: float,
):
    """
    Preprocess raw image extracting from dicom files.
    1. Apply formula:
        new_img = old_img * rescale_slope + rescale_intercept
    2. Apply HU window

    Args:
        img: input image (numpy array)
        ww: window width of HU window
        wc: window center of HU window
        rescale_slope: float
        rescale_intercept: float

    Return
    """
    img = apply_window(img, ww, wc)

    return img


def build_model(model_name):
    model = timm.create_model(model_name, pretrained=False)
    num_class = 4
    if "resnet" in model_name:
        model.fc = nn.Linear(512, num_class)
    elif "efficientnet" in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    return model


def load_model(model, path):
    if os.path.isfile(path):
        ckpt = torch.load(path, "cpu")
        model.load_state_dict(ckpt.pop("state_dict"))
        start_epoch, best_metric = ckpt["epoch"], ckpt["best_metric"]

    return model


class Data(Dataset):
    def __init__(self, imgs):
        """A Dataset object that load all data for running

        Args:
            cfg (CfgNode): Config object containing running configuration
            mode (str): Model running mode
        """
        self.imgs = imgs.transpose(2, 1, 0)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = cv2.flip(img, 0)
        img = preprocess_image(img, window_width, window_center).astype("float")
        img = cv2.normalize(
            img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        img = np.stack((img,) * 3, axis=-1)

        # RESIZE IMAGE
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        img = img.transpose(2, 0, 1)

        return img


def build_dataloader(imgs):
    """Build dataloader

    Returns:
        dataloader: Dataloader object
    """

    dataset = Data(imgs)
    # DEBUG: Only take a subset of dataloader to run script
    dataloader = DataLoader(
        dataset, 2, pin_memory=False, shuffle=False, drop_last=False, num_workers=4
    )
    return dataloader


def predict_scan(nii_path, model):

    img = nib.load(nii_path)
    img = img.get_fdata()

    dataloader = build_dataloader(img)
    preds = list()

    for image in tqdm(dataloader):
        with torch.no_grad():
            image = image.float().cuda()
            output = model(image)
        sigmoid = nn.Sigmoid()
        pred = torch.argmax(sigmoid(output), 1)

        # Convert target, prediction to numpy
        pred = list(pred.detach().cpu().numpy())
        preds += pred

    series_pred = mode(preds)

    del img, dataloader, preds, output
    return series_pred


if __name__ == "__main__":
    model = build_model(model_name)
    model_ckp = load_model(model, ckp_path)
    model.eval()
    model.cuda()

    predictions = []

    for scan in tqdm(data):
        predictions.append(predict_scan(scan, model))

    scan_names = [scan.split("/")[-1] for scan in data]
    df = pd.DataFrame({"Name": scan_names, "Prediction": predictions})
    df.to_csv("LITS_prediction.csv", index=False)
