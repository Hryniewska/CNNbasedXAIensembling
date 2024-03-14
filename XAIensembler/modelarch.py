#!/usr/bin/env python
# coding: utf-8

# **Importing Required Libraries**

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable as V


# **Context Encoder Architecture**


class DAC(nn.Module):
    """
    Dilated Atrous Convolution module.

    This module applies dilated atrous convolutions to the input tensor using multiple convolution layers.
    It is used as part of the context encoder architecture.

    Args:
        channels (int): Number of input and output channels.

    Returns:
        torch.Tensor: Transformed tensor after dilated atrous convolutions.
    """

    def __init__(self, channels):
        super(DAC, self).__init__()
        self.conv11 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=1, padding=1
        )
        self.relu1 = nn.ReLU()

        self.conv21 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=3, padding=3
        )
        self.conv22 = nn.Conv2d(
            channels, channels, kernel_size=1, dilation=1, padding=0
        )
        self.relu2 = nn.ReLU()

        self.conv31 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=1, padding=1
        )
        self.conv32 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=3, padding=3
        )
        self.conv33 = nn.Conv2d(
            channels, channels, kernel_size=1, dilation=1, padding=0
        )
        self.relu3 = nn.ReLU()

        self.conv41 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=1, padding=1
        )
        self.conv42 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=3, padding=3
        )
        self.conv43 = nn.Conv2d(
            channels, channels, kernel_size=3, dilation=5, padding=5
        )
        self.conv44 = nn.Conv2d(
            channels, channels, kernel_size=1, dilation=1, padding=0
        )
        self.relu4 = nn.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv11(x))

        c2 = self.conv21(x)
        c2 = self.relu2(self.conv22(c2))

        c3 = self.conv31(x)
        c3 = self.conv32(c3)
        c3 = self.relu3(self.conv33(c3))

        c4 = self.conv41(x)
        c4 = self.conv42(c4)
        c4 = self.conv43(c4)
        c4 = self.relu4(self.conv44(c4))

        c = x + c1 + c2 + c3 + c4

        return c


class RMP(nn.Module):
    """
    Residual Multi Kernel Pooling module.

    This module performs max pooling at different kernel sizes and combines the results with the input tensor
    to create a feature representation. It is used as part of the context encoder architecture.

    Args:
        channels (int): Number of input channels.

    Returns:
        torch.Tensor: Feature representation after residual multi kernel pooling.
    """

    def __init__(self, channels):
        super(RMP, self).__init__()

        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(channels, out_channels=1, kernel_size=1)

        self.max2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(channels, out_channels=1, kernel_size=1)

        self.max3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv3 = nn.Conv2d(channels, out_channels=1, kernel_size=1)

        self.max4 = nn.MaxPool2d(kernel_size=6)
        self.conv4 = nn.Conv2d(channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        m1 = self.max1(x)
        m1 = F.interpolate(self.conv1(m1), size=x.size()[2:], mode="bilinear")

        m2 = self.max2(x)
        m2 = F.interpolate(self.conv2(m2), size=x.size()[2:], mode="bilinear")

        m3 = self.max3(x)
        m3 = F.interpolate(self.conv3(m3), size=x.size()[2:], mode="bilinear")

        m4 = self.max4(x)
        m4 = F.interpolate(self.conv4(m4), size=x.size()[2:], mode="bilinear")

        m = torch.cat([m1, m2, m3, m4, x], axis=1)

        return m


class Decoder(nn.Module):
    """
    Decoder module.

    This module performs decoding operations by applying convolutions and transposed convolutions to the input tensor.
    It is used as part of the context encoder architecture.

    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters in the convolutional layers.

    Returns:
        torch.Tensor: Decoded tensor.
    """

    def __init__(self, in_channels, n_filters):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        return x


class CE_Net_(nn.Module):
    """
    Context Encoder Network architecture.

    This class defines the complete architecture of the Context Encoder Network, which includes DAC, RMP, and Decoder modules.
    It also implements the forward pass for the network.

    Args:
        params: Parameters for configuring the network.
        num_classes (int): Number of output classes.
        num_channels (int): Number of input channels.
        number_of_encoders (int): Number of times to repeat the encoder.

    Returns:
        tuple: Tuple containing the output tensor and ensemble explanations.
    """

    def __init__(self, params, num_classes=1, num_channels=3, number_of_encoders=1):
        super(CE_Net_, self).__init__()

        if params.aggregation_method == "channel":
            num_channels = len(params.XAI_methods) * 3
        elif (
            params.aggregation_method == "concat" or params.aggregation_method == "sum"
        ):
            number_of_encoders = len(params.XAI_methods)

        if params.aggregation_method == "concat":
            filters = [
                64 * len(params.XAI_methods),
                128 * len(params.XAI_methods),
                256 * len(params.XAI_methods),
                512 * len(params.XAI_methods),
            ]
        elif (
            params.aggregation_method == "sum" or params.aggregation_method == "channel"
        ):
            filters = [64, 128, 256, 512]

        self.conv1 = {}
        self.bn1 = {}
        self.maxpool1 = {}
        self.encoder1 = {}
        self.encoder2 = {}
        self.encoder3 = {}
        self.encoder4 = {}

        for i in range(number_of_encoders):
            resnet = models.resnet34(weights="IMAGENET1K_V1").to(params.device)

            layer = resnet.conv1
            new_layer = nn.Conv2d(
                in_channels=num_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias,
            )
            new_layer.weight[:, : layer.in_channels, :, :].data[...] = V(
                layer.weight.clone(), requires_grad=True
            )

            self.conv1[i] = new_layer.to(params.device)
            self.bn1[i] = resnet.bn1
            self.maxpool1[i] = resnet.maxpool

            self.encoder1[i] = resnet.layer1
            self.encoder2[i] = resnet.layer2
            self.encoder3[i] = resnet.layer3
            self.encoder4[i] = resnet.layer4

        self.dac = DAC(filters[3])
        self.rmp = RMP(filters[3])

        self.decoder4 = Decoder(filters[3] + 4, filters[2])
        self.decoder3 = Decoder(filters[2], filters[1])
        self.decoder2 = Decoder(filters[1], filters[0])
        self.decoder1 = Decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, params, expl):
        # repeat encoder same number of times as XAI methods
        e1, e2, e3, e4 = [], [], [], []

        if params.aggregation_method == "channel":
            iterator = [expl]
        elif (
            params.aggregation_method == "concat" or params.aggregation_method == "sum"
        ):
            iterator = torch.split(expl, 3, dim=1)

        for i, x in enumerate(iterator):
            # Encoder
            x = self.conv1[i](x)
            x = F.relu(self.bn1[i](x))
            x = self.maxpool1[i](x)

            x1 = self.encoder1[i](x)
            x2 = self.encoder2[i](x1)
            x3 = self.encoder3[i](x2)
            x4 = self.encoder4[i](x3)

            e1.append(x1)
            e2.append(x2)
            e3.append(x3)
            e4.append(x4)

        if params.aggregation_method == "concat":
            e1 = torch.cat(e1, 1)
            e2 = torch.cat(e2, 1)
            e3 = torch.cat(e3, 1)
            e4 = torch.cat(e4, 1)
        elif (
            params.aggregation_method == "sum" or params.aggregation_method == "channel"
        ):
            e1 = torch.sum(torch.stack(e1, dim=4), 4, keepdim=True).squeeze(4)
            e2 = torch.sum(torch.stack(e2, dim=4), 4, keepdim=True).squeeze(4)
            e3 = torch.sum(torch.stack(e3, dim=4), 4, keepdim=True).squeeze(4)
            e4 = torch.sum(torch.stack(e4, dim=4), 4, keepdim=True).squeeze(4)

        # Center
        e4 = self.dac(e4)
        e4 = self.rmp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = F.relu(self.finaldeconv1(d1))
        out = self.finalconv2(out)

        return torch.sigmoid(out), out


class dice_bce_loss(nn.Module):
    """
    Dice Coefficient binary cross entropy loss function.

    This class defines the combined Dice Coefficient and binary cross-entropy loss function.

    Args:
        batch (bool): Whether to compute the loss for a batch of data.

    Returns:
        torch.Tensor: Computed loss value.
    """

    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a


def acc_sen(pred, mask):
    """
    Calculate accuracy and sensitivity metrics.

    This function computes accuracy and sensitivity metrics based on predicted and actual masks.

    Args:
        pred (torch.Tensor): Predicted mask.
        mask (torch.Tensor): Actual mask.

    Returns:
        tuple: Tuple containing calculated accuracy, sensitivity, precision, F1 score, and IoU.
    """

    pred = torch.round(pred)
    TP = (mask * pred).sum(1).sum(1).sum(1)
    TN = ((1 - mask) * (1 - pred)).sum(1).sum(1).sum(1)
    FP = pred.sum(1).sum(1).sum(1) - TP
    FN = mask.sum(1).sum(1).sum(1) - TP
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc = torch.sum(acc)

    sen = TP / (TP + FN)
    sen = torch.sum(sen)

    prec = TP / (TP + FP)
    prec = torch.sum(prec)

    f1 = 2 * TP / (2 * TP + FP + FN)
    f1 = torch.sum(f1)

    iou = TP / (TP + FP + FN)
    iou = torch.sum(iou)

    return acc, sen, prec, f1, iou


class MyFrame:
    """
    Custom training framework class.

    This class provides a custom training framework for a given neural network model.

    Args:
        net (nn.Module): Neural network model.
        learning_rate (float): Learning rate for optimization.
        device (str): Device for computations (e.g., 'cuda' or 'cpu').
        evalmode (bool): Whether to enable evaluation mode.

    Methods:
        set_input(self, img_batch, mask_batch=None): Set input data for training or evaluation.
        optimize(self): Optimize the network parameters.
        calculate_loss(self): Calculate loss and predictions.
        save(self, path): Save the model's state dictionary to a file.
        load(self, path): Load the model's state dictionary from a file.
        update_lr(self, new_lr, factor=False): Update the learning rate of the optimizer.
    """

    def __init__(self, params, learning_rate, device, evalmode=False):
        self.net = CE_Net_(params).to(device)
        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=learning_rate
        )
        self.loss = dice_bce_loss().to(device)
        self.lr = learning_rate
        self.params = params

    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch

    def optimize(self):
        self.optimizer.zero_grad()
        pred, _ = self.net.forward(self.params, self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss, pred

    def calculate_loss(self):
        with torch.no_grad():
            pred, ensemble_expl = self.net.forward(self.params, self.img)
            loss = self.loss(self.mask, pred)
            return loss, pred, ensemble_expl

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print("update learning rate: %f -> %f" % (self.lr, new_lr))
        self.lr = new_lr
