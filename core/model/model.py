import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def GhostNet(in_channels):
    model = torch.hub.load("huawei-noah/ghostnet", "ghostnet_1x", pretrained=False)
    model.conv_stem = torch.nn.Conv2d(
        in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )

    return model


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

        # self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_final = nn.Linear(676, 4)

    # forward method
    # def forward(self, input, label):
    def forward(self, input):
        x = torch.cat([input, input], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        x = x.view(x.size()[0], -1)
        x = self.fc_final(x)
        return x


class SE_layer_3d(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(SE_layer_3d, self).__init__()

        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.globalAvgPool = nn.AdaptiveAvgPool3d((1))

    def forward(self, input_tensor):
        b, c, d, w, h = input_tensor.size()

        # Average along each channel
        squeeze_tensor = self.globalAvgPool(input_tensor).view(b, c).float()

        # Channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(b, c, 1, 1, 1))

        return output_tensor


class SE3D_net(nn.Module):
    def __init__(self, in_channels):
        super(SE3D_net, self).__init__()

        out_channels_1 = 64
        self.conv1 = nn.Conv3d(
            in_channels, out_channels_1, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels_1)
        self.relu = nn.ReLU(inplace=True)

        out_channels_2 = 256
        self.conv2 = nn.Conv3d(
            out_channels_1,
            out_channels_2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels_2)

        self.se = SE_layer_3d(out_channels_2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_semifinal = nn.Linear(out_channels_2, 512)
        self.fc_final = nn.Linear(out_channels_2, 4)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        se_layer = self.se(x)

        x = x.clone() + se_layer
        out = self.relu(x)

        out = self.global_pool(out)
        out = torch.squeeze(out)
        out = self.fc_final(out)

        return out


def build_model(cfg):
    model_name = cfg.MODEL.NAME
    pretrained = cfg.MODEL.PRETRAINED
    in_chans = cfg.DATA.INP_CHANNEL

    model = None
    if "resnet18" in model_name or "resnet34" in model_name:
        model = timm.create_model(
            model_name, in_chans=cfg.DATA.INP_CHANNEL, pretrained=False
        )
        model.fc = nn.Linear(512, cfg.MODEL.NUM_CLASSES)
    elif "efficientnet" in model_name:
        model = timm.create_model(
            model_name, in_chans=cfg.DATA.INP_CHANNEL, pretrained=False
        )
        model.classifier = nn.Linear(
            model.classifier.in_features, cfg.MODEL.NUM_CLASSES
        )
    elif "LightNet" in model_name:
        model = LightNet(in_channels=cfg.DATA.INP_CHANNEL)
    elif "3D" in model_name:
        model = SE3D_net(in_channels=1)
    elif "GhostNet" in model_name:
        model = GhostNet(in_channels=cfg.DATA.INP_CHANNEL)
    elif "Discriminator" in model_name:
        model = Discriminator()

    else:
        model = timm.create_model(
            model_name,
            in_chans=cfg.DATA.INP_CHANNEL,
            num_classes=cfg.MODEL.NUM_CLASSES,
            pretrained=False,
        )

    # # Freeze all layer and only train last Linear layer
    # for layer in model.modules():
    #     if not isinstance(layer, nn.Linear):
    #         for key, param in layer.named_parameters():
    #             param.requires_grad = False

    # for layer in model.modules():
    #     if isinstance(layer, nn.Linear):
    #         for key, param in layer.named_parameters():
    #             param.requires_grad = True

    return model
