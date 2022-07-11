import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import argparse
import os
import subprocess
import zipfile
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import random
from os import listdir
from os.path import isfile, join
import numpy as np
import csv
import pickle as pkl

from torch.autograd import Variable
import networkx as nx
from numpy import linalg as LA

import pandas as pd
from scipy import stats
import networkx as nx

import warnings

warnings.filterwarnings("ignore")


class Model(pl.LightningModule):
    def __init__(self, config, logger):
        super(Model, self).__init__()
        self.config = config
        self.n_logger = logger

        self.scales = torch.nn.Parameter(
            torch.randn(self.config["DATASET"]["NUM_SCALES"], 1)
        )
        torch.nn.init.uniform_(self.scales, a=0.5, b=1.5)

        self.scales.requires_grad = True

        self.model = nn.Sequential(
            nn.Linear(
                1
                * self.config["DATASET"]["NUM_ROI"]
                * self.config["DATASET"]["NUM_ROI"],
                256,
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=self.config["NET"]["DO"]),
            nn.Linear(256, self.config["DATASET"]["NUM_CLASSES"]),
        )

        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.batchnorm = nn.BatchNorm1d(
            1 * self.config["DATASET"]["NUM_ROI"] * self.config["DATASET"]["NUM_ROI"]
        )
        self.dropout = nn.Dropout(p=self.config["NET"]["DO"])

    def forward(self, x):

        torch.clamp(self.scales, 0.5, 1.5)

        eigvalues_x, eigvectors_x = (
            x[:, :, 0, :].view(-1, 1, self.config["DATASET"]["NUM_ROI"]),
            x[:, :, 1:, :],
        )

        scale_eigvalues_x = self.get_eigvalues_with_scales(
            eigvalues_x, self.scales.pow(2)
        )

        x = self.eig2L(scale_eigvalues_x, eigvectors_x)

        x = x.view(
            -1,
            1 * self.config["DATASET"]["NUM_ROI"] * self.config["DATASET"]["NUM_ROI"],
        )
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.model(x)

        return x, self.scales

    def get_eigvalues_with_scales(self, eigvalues, scales):

        eigvalues = eigvalues.permute(0, 2, 1)

        scales_repeat = scales.repeat(eigvalues.shape[0], 1)
        scales_repeat = scales_repeat.permute(1, 0).view(
            eigvalues.shape[0], 1, self.config["DATASET"]["NUM_SCALES"]
        )

        eigscales = torch.bmm(eigvalues, scales_repeat)
        eigscales = eigscales.permute(0, 2, 1)

        eigscales_exp_all = torch.exp(-eigscales)
        kernel_eigscales_band = torch.mul(eigscales, eigscales_exp_all).pow(2)
        kernel_eigscales_band = kernel_eigscales_band / 0.1353

        eigenvlaues_band = (
            eigvalues.permute(0, 2, 1)
            .view(-1, 1, self.config["DATASET"]["NUM_ROI"])
            .repeat(1, self.config["DATASET"]["NUM_SCALES"], 1)
        )

        kernels = (
            torch.mul(eigenvlaues_band, kernel_eigscales_band)
            .sum(dim=1)
            .unsqueeze(dim=1)
        )

        eigvalues_repeat = eigvalues.reshape(-1, 1).repeat(1, 1)
        eigvalues_repeat = eigvalues_repeat.permute(1, 0).reshape(
            -1, 1, self.config["DATASET"]["NUM_ROI"]
        )

        kernel_eigscales_values = torch.mul(eigvalues_repeat, kernels)

        return kernel_eigscales_values

    def eig2L(self, eigvalues, eigvectors):
        new_L = torch.from_numpy(
            np.zeros(
                (
                    1,
                    self.config["DATASET"]["NUM_ROI"],
                    self.config["DATASET"]["NUM_ROI"],
                ),
                dtype=np.float32,
            )
        ).to(self.device)

        for data_idx in range(eigvalues.shape[0]):

            each_eigvalues = eigvalues[data_idx]
            each_eigvectors = eigvectors[data_idx][0]

            for scale_idx in range(each_eigvalues.shape[0]):

                each_eigvalues_scale = torch.diag(each_eigvalues[scale_idx])

                each_UTA = torch.mm(each_eigvectors, each_eigvalues_scale)

                each_UTAU = torch.mm(each_UTA, torch.transpose(each_eigvectors, 0, 1))

                # each_UTAU = each_UTAU.view()

                new_L = torch.cat(
                    (
                        new_L,
                        each_UTAU.view(
                            1,
                            self.config["DATASET"]["NUM_ROI"],
                            self.config["DATASET"]["NUM_ROI"],
                        ),
                    ),
                    0,
                )

        return new_L[1:]

    def poly_lr_scheduler(
        self, optimizer, init_lr, iter, lr_decay_iter=1, max_iter=1000, power=0.9
    ):
        """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr * (1 - iter / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def configure_optimizers(self):

        optimizer = self.get_optim(self.config)
        # sch = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=self.config["TRAIN"]["LR_STEP_SIZE"],
        #     gamma=self.config["TRAIN"]["LR_GAMMA"],
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
        # }
        return optimizer

    def get_optim(self, config):

        if not hasattr(torch.optim, config["NET"]["OPT"]):
            print("Optimiser {} not supported".format(config["NET"]["OPT"]))
            raise NotImplementedError

        optim = getattr(torch.optim, config["NET"]["OPT"])

        if config["NET"]["OPT"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config["NET"]["LR"],
                betas=(config["NET"]["BETA1"], 0.999),
                weight_decay=config["NET"]["WEIGHT_DECAY"],
            )
        elif config["NET"]["OPT"] == "SGD":
            print(
                "Using SGD >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(
                    config["NET"]["LR"],
                    config["NET"]["MOMENTUM"],
                    config["NET"]["WEIGHT_DECAY"],
                )
            )

            optimizer = torch.optim.SGD(
                # [
                #     # {"params": self.decode_head.parameters()},
                #     # {"params": self.aux_head.parameters()},
                #     # {"params": self.aspp_classifier.parameters()},
                #     # {"params": self.aux_classifier.parameters()},
                #     # {
                #     #     "params": self.backbone.parameters(),
                #     #     "lr": self.config["NET"]["BACKBONE_LR"],
                #     # },
                #     {"params": self.parameters()}
                # ],
                [
                    {"params": self.model.parameters()},
                    {"params": self.scales, "lr": self.config["NET"]["SCALE_LR"]},
                ],
                lr=config["NET"]["LR"],
                momentum=config["NET"]["MOMENTUM"],
                weight_decay=config["NET"]["WEIGHT_DECAY"],
            )

        else:
            optimizer = optim(self.parameters(), lr=config["NET"]["LR"])

        optimizer.zero_grad()

        return optimizer

    def loss_function(self, logits, labels):

        loss = self.focal_loss(logits, labels)

        return loss

    def focal_loss(self, inputs, targets):

        self.alpha = Variable(torch.ones(self.config["DATASET"]["NUM_CLASSES"], 1))
        self.gamma = 2
        self.class_num = self.config["DATASET"]["NUM_CLASSES"]
        self.size_average = True

        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, step_scales = self.forward(x)

        train_prediction = logits.argmax(1)

        train_acc = train_prediction.eq(y).view(-1).float().mean()

        loss = self.loss_function(logits, y)

        l1_reg = torch.tensor(0.0).to(self.device)
        for name, param in self.named_parameters():
            if name == "fc1.weight":
                l1_reg += torch.norm(param, p=1)
        loss += self.config["NET"]["L1"] * l1_reg

        scale_reg = torch.tensor(0.0).to(self.device)
        for name, param in self.named_parameters():
            if name == "scales":
                for i in range(param.shape[0]):
                    self.log(
                        "scale" + str(i), param[i] ** 2, on_step=False, on_epoch=True
                    )
                scale_reg += torch.norm(param[0])
        loss += self.config["NET"]["SCALE_REG"] * scale_reg

        self.log("l1_reg", l1_reg, on_step=False, on_epoch=True)
        self.log("scale_reg", scale_reg, on_step=False, on_epoch=True)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        x, y = batch

        logits, step_scales = self.forward(x)

        val_prediction = logits.argmax(1)
        val_acc = val_prediction.eq(y).view(-1).float().mean()

        val_loss = self.loss_function(logits, y)

        l1_reg = torch.tensor(0.0).to(self.device)
        for name, param in self.named_parameters():
            if name == "fc.weight":
                l1_reg += torch.norm(param, p=1)
        val_loss += self.config["NET"]["L1"] * l1_reg

        scale_reg = torch.tensor(0.0).to(self.device)
        for name, param in self.named_parameters():
            if name == "scales":
                scale_reg += torch.norm(param)
        val_loss += self.config["NET"]["SCALE_REG"] * scale_reg

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
