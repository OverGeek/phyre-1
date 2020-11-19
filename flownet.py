# %%
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from phyre_rolllout_collector import load_phyre_rollouts, collect_solving_dataset
from cv2 import resize, imshow, waitKey
import cv2
from phyre_utils import draw_ball, grow_action_vector, vis_batch, make_mono_dataset, action_delta_generator, gifify, \
    vis_pred_path, vis_pred_path_task, make_mono_dataset_2, make_mono_dataset_sequential
from itertools import chain
import argparse
import os
import random
import phyre
from dijkstra import find_distance_map_obj
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import gzip
from tqdm import tqdm
import pandas as pd
import pickle
from collections import deque


# %%


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert (input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispNetS(nn.Module):

    def __init__(self, alpha=10, beta=0.01, inp_channels=5):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(inp_channels, conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6], upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1


class SpatialConv(nn.Module):
    def __init__(self, conv, direction, inplace=True, trans=False):
        super().__init__()
        self.inplace = inplace
        self.conv = conv
        self.direction = direction
        self.trans = trans

    def forward(self, X):
        sideways = False
        direction = self.direction
        if direction == "right":
            sideways = True
            pos = 0
            end = X.shape[3] - 1
            add = 1
        elif direction == "left":
            sideways = True
            end = X.shape[3] - 1 if self.trans else 0
            pos = 0 if self.trans else X.shape[3] - 1
            add = -1
        elif direction == "up":
            end = X.shape[2] - 1 if self.trans else 0
            pos = 0 if self.trans else X.shape[2] - 1
            add = -1
        elif direction == "down":
            pos = 0
            end = X.shape[2] - 1
            add = 1
        else:
            print("Direction not understood!")
            return X

        # ORIGINAL "INPLACE" CONCEPT from paper
        if self.inplace:

            if self.trans:
                if add < 0:
                    X = X.flip(-1).clone() if sideways else X.flip(-2).clone()
                else:
                    X = X.clone()

                # transpose if going sideways
                X = X.transpose(-2, -1) if sideways else X

                # stepping through the tensor
                while pos != end:
                    # print(X[:,:,pos,:].shape, X[:,:,pos-kd:pos+1,:].shape, self.conv(X[:,:,pos-kd:pos+1,:]).shape)
                    X[:, :, pos + 1, :] += T.tanh(self.conv(X[:, :, pos, :]))
                    pos += 1

                # re-flip if direction was 'negative'
                if add < 0:
                    X = X.flip(-1) if sideways else X.flip(-2)

                # re-transpose if sideways
                return X.transpose(-2, -1) if sideways else X

            else:
                X = X.clone()
                while pos != end:
                    if sideways:
                        X[:, :, :, pos + add] += T.tanh(self.conv(X[:, :, :, pos]))
                    else:
                        X[:, :, pos + add, :] += T.tanh(self.conv(X[:, :, pos, :]))
                    pos += add
                return X

        # Alternative concept
        else:
            if sideways:
                first = T.tanh(self.conv(X[:, :, :, pos]))
                Y = T.zeros(X.shape[0], first.shape[1], X.shape[2], first.shape[2])
                Y[:, :, :, pos] = first
            else:
                first = T.tanh(self.conv(X[:, :, pos, :]))
                Y = T.zeros(X.shape[0], first.shape[1], first.shape[2], X.shape[3])
                Y[:, :, pos, :] = first
            while pos != end:
                if sideways:
                    Y[:, :, :, pos + add] += Y[:, :, :, pos] + T.tanh(self.conv(X[:, :, :, pos + add]))
                else:
                    Y[:, :, pos + add, :] += Y[:, :, pos, :] + T.tanh(self.conv(X[:, :, pos + add, :]))
                pos += add
            return Y


class SequentialConv(nn.Module):
    def __init__(self, conv, direction, inplace=True, trans=False):
        super().__init__()
        self.inplace = inplace
        self.conv = conv
        self.direction = direction

    def forward(self, X):
        # print('input shape', X.shape)
        # print('conv weight shape', self.conv.weight.shape)
        sideways = False
        direction = self.direction
        if direction == "right":
            sideways = True
            end = X.shape[3]
            pos = 0
            add = 1
        elif direction == "left":
            sideways = True
            end = -1
            pos = X.shape[3] - 1
            add = -1
        elif direction == "up":
            end = -1
            pos = X.shape[2] - 1
            add = -1
        elif direction == "down":
            end = X.shape[2]
            pos = 0
            add = 1
        else:
            print("Direction not understood!")
            return X

        # ORIGINAL "INPLACE" CONCEPT from paper
        if self.inplace:
            kd = self.conv.kernel_size[1] - 1 if sideways else self.conv.kernel_size[0] - 1
            # print(self.conv.kernel_size)
            pos += kd * add
            X = X.clone()
            while pos != end:
                # print(pos)
                if sideways:
                    if add < 0:
                        # print(X[:,:,:,pos].shape, X[:,:,:,pos:pos+kd+1].shape, self.conv(X[:,:,:,pos:pos+kd+1]).shape)
                        X[:, :, :, pos] += T.tanh(self.conv(X[:, :, :, pos:pos + kd + 1])[:, :, 0])
                    else:
                        # print(X[:,:,:,pos].shape, X[:,:,:,pos-kd:pos+1].shape, self.conv(X[:,:,:,pos-kd:pos+1]).shape)
                        X[:, :, :, pos] += T.tanh(self.conv(X[:, :, :, pos - kd:pos + 1])[:, :, 0])

                else:
                    if add < 0:
                        # print(X[:,:,pos,:].shape, X[:,:,pos:pos+kd+1,:].shape, self.conv(X[:,:,pos:pos+kd+1,:]).shape)
                        X[:, :, pos, :] += T.tanh(self.conv(X[:, :, pos:pos + kd + 1, :])[:, :, 0])
                    else:
                        # print(X[:,:,pos,:].shape, X[:,:,pos-kd:pos+1,:].shape, self.conv(X[:,:,pos-kd:pos+1,:]).shape)
                        X[:, :, pos, :] += T.tanh(self.conv(X[:, :, pos - kd:pos + 1, :])[:, :, 0])

                pos += add
            return X

        # Alternative concept
        else:
            kw = (self.conv.kernel_size[1] if sideways else self.conv.kernel_size[0]) - 1
            pos += kw * add
            if sideways:
                first = T.tanh(self.conv(X[:, :, :, pos + (kw * add):pos]))
                Y = T.zeros(X.shape[0], first.shape[1], X.shape[2], first.shape[2])
                Y[:, :, :, pos] = first
            else:
                first = T.tanh(self.conv(X[:, :, pos + (kw * add):pos, :]))
                Y = T.zeros(X.shape[0], first.shape[1], first.shape[2], X.shape[3])
                Y[:, :, pos, :] = first
            while pos != end:
                if sideways:
                    Y[:, :, :, pos + add] += Y[:, :, :, pos] + T.tanh(self.conv(X[:, :, :, pos + add]))
                else:
                    Y[:, :, pos + add, :] += Y[:, :, pos, :] + T.tanh(self.conv(X[:, :, pos + add, :]))
                pos += add
            return Y


class ImagiNet(nn.Module):
    def __init__(self):
        super().__init__()
        chs = 16
        # self.feature_dims = feature_dims
        # self.embed_dims = embed_dims
        # self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        conv_fn = lambda: nn.Conv1d(chs, chs, 5, 1, 2)
        self.r1 = SpatialConv(conv_fn(), "right", inplace=True)
        self.l1 = SpatialConv(conv_fn(), "left", inplace=True)
        self.u1 = SpatialConv(conv_fn(), "up", inplace=True)
        self.d1 = SpatialConv(conv_fn(), "down", inplace=True)
        self.r2 = SpatialConv(conv_fn(), "right", inplace=True)
        self.l2 = SpatialConv(conv_fn(), "left", inplace=True)
        self.u2 = SpatialConv(conv_fn(), "up", inplace=True)
        self.d2 = SpatialConv(conv_fn(), "down", inplace=True)
        self.init_conv = nn.Sequential(
            nn.Conv2d(7, chs, 3, 1, 1),
            nn.Tanh())
        self.end_conv = nn.Sequential(
            nn.Conv2d(chs, 1, 3, 1, 1),
            nn.Sigmoid())
        self.mid_conv = nn.Sequential(
            nn.Conv2d(7, 7, 3, 1, 1),
            nn.Tanh())
        self.big_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, 4, 2, 1),
            nn.LeakyReLU(0.1))
        self.deconv = nn.Sequential(
            # nn.Linear(embed_dims, 8*self.final_conv_width**2),
            nn.Tanh(),
            # View((-1, 8, self.final_conv_width, self.final_conv_width)),
            nn.ConvTranspose2d(8, 16, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, X):
        X = self.init_conv(X)
        for n in range(1):
            X = self.u1(X)
            X = self.d1(X)
            X = self.l1(X)
            X = self.r1(X)
            # X = self.u2(X)
            # X = self.d2(X)
            # X = self.l2(X)
            # X = self.r2(X)
            # X = self.mid_conv(X)
        X = self.end_conv(X)
        # X = F.relu(X[:,None,0])
        return X


class FlowNet(nn.Module):
    def __init__(self, in_dim, chs, sequ=False, trans=False):
        super().__init__()
        # self.feature_dims = feature_dims
        # self.embed_dims = embed_dims
        # self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        vertical_conv_fn = (lambda: nn.Conv2d(chs, chs, (2, 5), 1, (0, 2))) if sequ else (
            lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        horizontal_conv_fn = (lambda: nn.Conv2d(chs, chs, (5, 2), 1, (2, 0))) if sequ else (
            lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        flow_model = SequentialConv if sequ else SpatialConv
        self.r1 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l1 = flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u1 = flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d1 = flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
        self.r2 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l2 = flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u2 = flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d2 = flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_dim, chs, 3, 1, 1),
            nn.Tanh())
        self.end_conv = nn.Sequential(
            nn.Conv2d(chs, 1, 3, 1, 1),
            nn.Sigmoid())
        self.mid_conv = nn.Sequential(
            nn.Conv2d(7, 7, 3, 1, 1),
            nn.Tanh())
        self.big_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, 4, 2, 1),
            nn.LeakyReLU(0.1))
        self.deconv = nn.Sequential(
            # nn.Linear(embed_dims, 8*self.final_conv_width**2),
            nn.Tanh(),
            # View((-1, 8, self.final_conv_width, self.final_conv_width)),
            nn.ConvTranspose2d(8, 16, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, X):
        X = self.init_conv(X)
        for n in range(1):
            X = self.u1(X)
            X = self.d1(X)
            X = self.l1(X)
            X = self.r1(X)
            # X = self.u2(X)
            # X = self.d2(X)
            # X = self.l2(X)
            # X = self.r2(X)
            # X = self.mid_conv(X)
        X = self.end_conv(X)
        # X = F.relu(X[:,None,0])
        return X


class UpFlowNet(nn.Module):
    def __init__(self, in_dim, chs, sequ=False, trans=False):
        super().__init__()
        # self.feature_dims = feature_dims
        # self.embed_dims = embed_dims
        # self.final_conv_width = ((int(self.feature_dims**0.5)//4)-2)
        vertical_conv_fn = (lambda: nn.Conv2d(chs, chs, (2, 5), 1, (0, 2))) if sequ else (
            lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        horizontal_conv_fn = (lambda: nn.Conv2d(chs, chs, (5, 2), 1, (2, 0))) if sequ else (
            lambda: nn.Conv1d(chs, chs, 5, 1, 2))
        flow_model = SequentialConv if sequ else SpatialConv
        self.r1 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l1 = flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u1 = flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d1 = flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
        self.r2 = flow_model(horizontal_conv_fn(), "right", inplace=True, trans=trans)
        self.l2 = flow_model(horizontal_conv_fn(), "left", inplace=True, trans=trans)
        self.u2 = flow_model(vertical_conv_fn(), "up", inplace=True, trans=trans)
        self.d2 = flow_model(vertical_conv_fn(), "down", inplace=True, trans=trans)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_dim, chs, 3, 1, 1),
            nn.Tanh())
        self.end_conv = nn.Sequential(
            nn.Conv2d(chs, 1, 3, 1, 1),
            nn.Sigmoid())
        self.mid_conv = nn.Sequential(
            nn.Conv2d(7, 7, 3, 1, 1),
            nn.Tanh())
        self.big_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, 4, 2, 1),
            nn.LeakyReLU(0.1))
        self.deconv = nn.Sequential(
            # nn.Linear(embed_dims, 8*self.final_conv_width**2),
            nn.Tanh(),
            # View((-1, 8, self.final_conv_width, self.final_conv_width)),
            nn.ConvTranspose2d(8, 16, 3, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, X):
        X = self.init_conv(X)
        for n in range(1):
            X = self.u1(X)
            # X = self.d1(X)
            X = self.l1(X)
            X = self.r1(X)
            X = self.u2(X)
            # X = self.d2(X)
            X = self.l2(X)
            X = self.r2(X)
            # X = self.mid_conv(X)
        X = self.end_conv(X)
        # X = F.relu(X[:,None,0])
        return X


class ActionBallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.Conv2d(4, 1, 5, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)


class Discriminator(nn.Module):
    def __init__(self, in_dim, wid):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 4, 4, 2, 1),
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(4*4*16, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1))
        """
        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(2 ** (2 + i), 2 ** (3 + i), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, 8, 4, 2, 1), acti()] + [acti() if i % 2 else convs[i // 2] for i in
                                                             range(2 * len(folds))]
        reason = [nn.Flatten(), nn.Linear(2 ** (3 + max(folds)), 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid()]
        modules = encoder + reason
        self.model = nn.Sequential(*modules)

    def forward(self, X):
        return self.model(X)


class Pyramid(nn.Module):
    def __init__(self, in_dim, chs, wid, hidfac):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, chs, 4, 2, 1),
            nn.Sigmoid()
        )
        """

        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(int(2 ** (2 + i) * hidfac), int(2 ** (3 + i) * hidfac), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, int(8 * hidfac), 4, 2, 1), acti()] + [acti() if i % 2 else convs[i // 2] for i in
                                                                           range(2 * len(folds))]
        trans_convs = [nn.ConvTranspose2d(int(2 ** (3 + i) * hidfac), int(2 ** (2 + i) * hidfac), 4, 2, 1) for i in
                       reversed(folds)]
        decoder = [acti() if i % 2 else trans_convs[i // 2] for i in range(2 * len(folds))] + [
            nn.ConvTranspose2d(int(8 * hidfac), chs, 4, 2, 1), nn.Sigmoid()]
        modules = encoder + decoder
        self.model = nn.Sequential(*modules)
        # print(self.model.state_dict().keys())

        """
        convs = [(2**(2+i), 2**(3+i)) for i in folds]
        trans_convs = [(2**(3+i), 2**(2+i)) for i in reversed(folds)]
        print(convs)
        print(trans_convs)
        encoder = [(in_dim,8), 'acti'] + [f"acti" if i%2 else convs[i//2] for i in range(2*len(folds))]
        print(encoder)
        decoder = [f"acti" if i%2 else trans_convs[i//2] for i in range(2*len(folds))] + [(8,chs), 'Sigmoid']
        print(decoder)
        print(*(encoder+decoder), sep='\n')
        """

    def forward(self, X):
        return self.model(X)


class SmartPyramid(nn.Module):
    def __init__(self, in_dim, chs, wid):
        super().__init__()
        """
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, 8, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, chs, 4, 2, 1),
            nn.Sigmoid()
        )
        """

        folds = range(1, int(np.math.log2(wid)))
        acti = nn.ReLU
        convs = [nn.Conv2d(2 ** (2 + i), 2 ** (3 + i), 4, 2, 1) for i in folds]
        encoder = [nn.Conv2d(in_dim, 8, 4, 2, 1), acti()] + [acti() if i % 2 else convs[i // 2] for i in
                                                             range(2 * len(folds))]
        trans_convs = [nn.ConvTranspose2d(2 ** (3 + i), 2 ** (2 + i), 4, 2, 1) for i in reversed(folds)]
        decoder = [acti() if i % 2 else trans_convs[i // 2] for i in range(2 * len(folds))] + [
            nn.ConvTranspose2d(8, chs, 4, 2, 1), nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        encoding_dim = 2 ** (3 + max(folds))
        self.reason = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Tanh()
        )

    def forward(self, X):
        X = self.encoder(X)
        shape = X.shape
        X = self.reason(X.view(shape[0], shape[1]))
        X = self.decoder(X.view(*shape))
        return X


class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, wid):
        super().__init__()
        self.out_dim = out_dim
        acti = nn.ReLU
        self.wid = wid
        self.model = nn.Sequential(
            nn.Linear(in_dim * self.wid * self.wid, 512),
            acti(),
            nn.Linear(512, 256),
            acti(),
            nn.Linear(256, 128),
            acti(),
            nn.Linear(128, 256),
            acti(),
            nn.Linear(256, self.wid * self.wid * self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X.view(X.shape[0], -1)).view(-1, self.out_dim, self.wid, self.wid)


def obj_in_scene(X):
    ret = True if np.max(X) == 1 else False
    return ret


class FlownetSolver():
    def __init__(self, path: str, modeltype: str, width: int, eval_only=False, smart=False, run='', num_seeds=1,
                 device="cuda", hidfac=1, dijkstra=False, viz=100):
        super().__init__()
        self.device = ("cuda" if T.cuda.is_available() else "cpu") if device == "cuda" else "cpu"
        print("device:", self.device)
        self.path = path
        self.run = run
        self.width = width
        self.modeltype = modeltype
        self.discr = None
        self.r_fac = 1
        self.models = dict()
        self.num_seeds = num_seeds
        # self.cache = phyre.get_default_100k_cache('ball')
        self.cache = None
        self.hidfac = hidfac
        self.dijkstra = dijkstra
        self.viz = 100
        self.logger = dict()

        pyramid = SmartPyramid if smart else Pyramid

        if modeltype == "linear":
            self.models["tar_net"] = FullyConnected(5, 1, width)
            self.models["base_net"] = FullyConnected(5, 1, width)
            self.models["act_net"] = FullyConnected(7, 1, width)
            self.models["ext_net"] = FullyConnected(7, 1, width)
        elif modeltype == "pyramid":
            self.models["tar_net"] = pyramid(5 + int(self.dijkstra), 1, width, hidfac)
            self.models["base_net"] = pyramid(5, 1, width, hidfac)
            self.models["act_net"] = pyramid(7, 1, width, hidfac)
            self.models["ext_net"] = pyramid(7, 1, width, hidfac)
        elif modeltype == "scnn":
            self.models["tar_net"] = FlowNet(5 + int(self.dijkstra), 16, sequ=True, trans=False)
            self.models["base_net"] = FlowNet(5, 16, sequ=True, trans=False)
            self.models["act_net"] = FlowNet(7, 16, sequ=True, trans=False)
            self.models["ext_net"] = UpFlowNet(7, 16, sequ=True)
        elif modeltype == "brute":
            self.models["tar_net"] = pyramid(8 + int(self.dijkstra), 1, width, hidfac)
            self.models["base_net"] = pyramid(6, 1, width, hidfac)
            self.models["act_net"] = pyramid(7, 1, width, hidfac)
            self.models["ext_net"] = Discriminator(9, width)
        elif modeltype == "combi":
            self.models["tar_net"] = pyramid(5 + int(self.dijkstra), 1, width, hidfac)
            self.models["base_net"] = pyramid(5, 1, width, hidfac)
            self.models["act_net"] = pyramid(7, 1, width, hidfac)
            self.models["ext_net"] = pyramid(7, 1, width, hidfac)
            self.models["sim_net"] = pyramid(6, 1, width, hidfac)
            self.models["comb_net"] = pyramid(6, 3, width, hidfac)
            self.models["success_net"] = Discriminator(9, width)
        elif modeltype == "skip_pyramid":
            self.models["tar_net1"] = pyramid(5 + int(self.dijkstra), 1, width, hidfac)
            self.models["base_net"] = pyramid(5, 1, width, hidfac)
            self.models["act_net"] = pyramid(7, 1, width, hidfac)
            self.models["ext_net"] = pyramid(7, 1, width, hidfac)
            self.models["tar_net2"] = pyramid(6 + int(self.dijkstra), 1, width, hidfac)
        elif modeltype == "SfMLearner":
            self.models["SfM1"] = DispNetS(alpha=1.0, beta=0.0, inp_channels=3)
            self.models["SfM2"] = DispNetS(alpha=1.0, beta=0.0, inp_channels=5)
            self.models["SfM_sequential"] = DispNetS(alpha=1.0, beta=0.0, inp_channels=21)

        else:
            print("ERROR modeltype not understood", modeltype)

        print("succesfully initialized models")

    def resize_tensor(self, X, size):
        X_resized = F.interpolate(X[:, None, :, :], size=(size, size), mode='bicubic')

        return X_resized.to(self.device)

    def cut_off(self, tensor, limit=1):
        tensor[tensor > limit] = limit
        tensor[tensor < 0] = 0
        return tensor

    def get_actions(self, tasks, init_scenes, brute=False):
        if brute:
            return self.brute_searched_actions(tasks, init_scenes)
        else:
            return self.generative_actions(tasks, init_scenes)

    def get_proposals(self, tasks, name):
        sim = phyre.initialize_simulator(tasks, 'ball')
        all_initial_scenes = T.tensor(
            [[cv2.resize((scene == channel).astype(float), (self.width, self.width)) for channel in range(2, 7)] for
             scene in sim.initial_scenes]).float().flip(-2)
        task_dict = dict()

        self.to_train()
        actions = np.zeros((len(tasks), 3))
        pipelines = []
        num_batches = 1 + len(tasks) // 64
        # BACTHING ALL TASKS
        for batch in range(num_batches):
            init_scenes = all_initial_scenes[batch * 64:64 * (batch + 1)].to(self.device)
            # emb_init_scenes = F.embedding(T.tensor(batch_scenes), T.eye(phyre.NUM_COLORS)).transpose(-1,-3)[:,1:6].float()
            # print(init_scenes.equal(emb_init_scenes))

            # FORWARD PASS
            with T.no_grad():
                base_paths = self.models["base_net"](init_scenes)
                target_paths = self.models["tar_net"](init_scenes)
                action_paths = self.models["act_net"](T.cat((init_scenes, target_paths, base_paths), dim=1))
                action_balls = self.models["ext_net"](T.cat((init_scenes, target_paths, action_paths), dim=1))
                # print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, action_balls), dim=1)
                # text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred']
                # vis_batch(print_batch, f'result/flownet/solver/{self.path}', f'{batch}', text=text)

            batch_tasks = tasks[64 * batch:64 * (batch + 1)]
            os.makedirs(f'result/solver/generative/', exist_ok=True)
            # LOOPING THROUGH ALL TASKS IN BATCH
            for idx, ball in enumerate(action_balls[:, 0].cpu()):
                task = batch_tasks[idx]
                print("generating proposals for task", task)
                # CHOOSE ONE VECTOR EXTRACTION METHOD
                # a = pic_to_action_vector(ball, r_fac=1.5)
                mask = np.max(init_scenes[idx].cpu().numpy(), axis=0)
                # print(mask.shape)
                a = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                       updates=5)
                a2 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a3 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a4 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a5 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                # print(a)

                drawn = draw_ball(self.width, a[0], a[1], a[2], invert_y=True).to(self.device)
                drawn2 = draw_ball(self.width, a2[0], a2[1], a2[2], invert_y=True).to(self.device)
                drawn3 = draw_ball(self.width, a3[0], a3[1], a3[2], invert_y=True).to(self.device)
                drawn4 = draw_ball(self.width, a4[0], a4[1], a4[2], invert_y=True).to(self.device)
                drawn5 = draw_ball(self.width, a5[0], a5[1], a5[2], invert_y=True).to(self.device)

                def get_nice_xyr(a):
                    x, y, r = str(round(a[0], 2)), str(round(a[1], 2)), str(round(a[2], 2))
                    return f"{x} {y} {r}"

                back = init_scenes[None, idx, 3:].sum(dim=1)[:, None]
                back = back / max(back.max(), 1)
                inits = init_scenes[None, idx, None]
                vis_line = T.cat((
                    T.stack((back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),
                    # inital scene
                    T.stack((back, base_paths[idx, None] + back, inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),
                    # scene with base
                    T.stack((action_paths[idx, None] + back, inits[:, :, 0] + target_paths[idx, None] + back,
                             inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),  # scene with action path and target
                    T.stack(
                        (action_balls[idx, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                        dim=-1),  # scene with action ball
                    T.stack((drawn[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn2[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn3[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn4[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn5[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1)),
                    dim=1).detach()
                vis_line = self.cut_off(vis_line.cpu())
                white = T.ones_like(vis_line)
                white[:, :, :, :, [0, 1]] -= vis_line[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [0, 2]] -= vis_line[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [1, 2]] -= vis_line[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                vis_line = self.cut_off(white)
                text = ['initial\nscene', 'scene with\nbase path\nprediction', 'action and\ntarget path\nprediction',
                        'scene with\naction ball\nprediction', f'injected x,y,r\naction vector\n{get_nice_xyr(a)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a2)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a3)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a4)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a5)}']
                vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f"seeds", text=text,
                          save=True, font_size=9)
                vis_line = vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f"seeds",
                                     text=text, save=False, font_size=9)

                # pipelines.append(vis_line)

                # Radius times 4 since actions are scaled this way (and times 2 to get diameter??)
                a[2] = a[2] * 4 * 2
                a2[2] = a2[2] * 4 * 2
                a3[2] = a3[2] * 4 * 2
                a4[2] = a4[2] * 4 * 2
                a5[2] = a5[2] * 4 * 2

                # print(a)
                # saving action
                base_actions = [a, a2, a3, a4, a5]
                actions[idx + batch * 64] = a
                delta_generator = action_delta_generator(pure_noise=True)

                # MAKE MORE ACTIONS
                tried_actions = []
                num_proposals = 100
                action_proposals = np.zeros((5, num_proposals, 3))
                task_actions = np.zeros((5 * num_proposals, 3))
                for seed_id in range(5):
                    # print(task, "seed",seed_id)
                    for proposal_idx in range(num_proposals):
                        action = base_actions[seed_id]
                        tmp_base_action = action
                        while self.similar_action_tried(action, tried_actions) or self.is_invalid(action, mask):
                            delta = delta_generator.__next__()
                            action = np.clip(action + delta, 0, 1)
                        # print("delta", action-tmp_base_action)
                        tried_actions.append(action)
                        action_proposals[seed_id, proposal_idx] = action
                    task_actions[seed_id::5] = action_proposals[seed_id]

                task_dict[task] = task_actions
        return task_dict

    def generative_actions(self, tasks, initial_scenes):
        # self.to_eval()
        actions = np.zeros((len(tasks), 3))
        num_batches = 1 + len(tasks) // 64
        for batch in range(num_batches):
            init_scenes = initial_scenes[batch * 64:64 * (batch + 1)]
            # emb_init_scenes = F.embedding(T.tensor(batch_scenes), T.eye(phyre.NUM_COLORS)).transpose(-1,-3)[:,1:6].float()
            # print(init_scenes.equal(emb_init_scenes))
            with T.no_grad():
                base_paths = self.models["base_net"](init_scenes)
                target_paths = self.models["tar_net"](init_scenes)
                action_paths = self.models["act_net"](T.cat((init_scenes, target_paths, base_paths), dim=1))
                action_balls = self.models["ext_net"](T.cat((init_scenes, target_paths, action_paths), dim=1))
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, action_balls), dim=1)
                # text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred']
                # vis_batch(print_batch, f'result/flownet/solver/{self.path}', f'{batch}', text=text)
            batch_task = tasks[64 * batch:64 * (batch + 1)]
            os.makedirs(f'result/solver/generative/', exist_ok=True)
            for idx, ball in enumerate(action_balls[:, 0]):
                task = batch_task[idx]

                # CHOOSE ONE VECTOR EXTRACTION METHOD
                # a = pic_to_action_vector(ball, r_fac=1.5)
                a = grow_action_vector(ball, r_fac=self.r_fac, check_border=True)
                print(a)

                drawn = draw_ball(self.width, *a, invert_y=True)
                pure_scene = T.as_tensor(np.max(print_batch[idx, [0, 1, 2, 3, 4]].numpy(), axis=0))
                scene_with_estimate = T.as_tensor(np.max(print_batch[idx, [0, 1, 2, 3, 4, -1]].numpy(), axis=0))
                scene_with_injected = T.as_tensor(np.max(T.stack((scene_with_estimate, drawn), dim=0).numpy(), axis=0))
                init_with_injected = T.as_tensor(np.max(T.cat((init_scenes[idx], drawn[None]), dim=0).numpy(), axis=0))
                # print(print_batch[idx].shape)
                # print(scene_with_estimate[None].shape)
                # print(scene_with_injected[None].shape)
                # print(init_with_injected[None].shape)
                pipeline = T.cat((print_batch[idx], scene_with_estimate[None], pure_scene[None],
                                  scene_with_injected[None], init_with_injected[None]), dim=0)

                text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'full\nscene',
                        'base\npred', 'target\npred', 'action\npred', 'a-ball\npred', 'action\nestimate',
                        'estimate\n/w action', '   final\n   action']
                x, y, r = round(a[0], 3), round(a[1], 3), round(a[2], 3)
                vis_batch(pipeline[None], f'result/solver/generative', f"{task}__{str(a)}", text=text)
                # plt.imsave(f'result/solver/pyramid/{task}___{(x,y,r)}.png', draw_ball(32, *a, invert_y = True))
                # Radius times 4
                a[2] = a[2] * 4
                # img = np.max(print_batch[idx,[0,1,2,3,4,-1]].numpy(), axis=0)
                # plt.imsave(f'result/solver/pyramid/{task}__{str(a)}.png', img)
                print(a)
                actions[idx + batch * 64] = a

        # print(list(zip(tasks,actions)))

        return actions

    def similar_action_tried(self, action, tries):
        for other in tries:
            if (np.linalg.norm(action - other) < 0.02):
                print("similiar action already tried", action, other, end="\r")
                return True
        return False

    def is_invalid(self, action, mask):
        x, y, d = action
        r = d / 8
        # print(x,y,d)
        overlap = mask[draw_ball(self.width, x, y, r, invert_y=True) > 0].any()

        return (x - r < 0) or (x + r > 1) or (y - r < 0) or (y + r > 1) or (d > 1) or (d < 0) or overlap

    def generative_auccess(self, tasks, name, pure_noise=False):
        sim = phyre.initialize_simulator(tasks, 'ball')
        all_initial_scenes = T.tensor(
            [[cv2.resize((scene == channel).astype(float), (self.width, self.width)) for channel in range(2, 7)] for
             scene in sim.initial_scenes]).float().flip(-2)
        eva = phyre.Evaluator(tasks)
        cache = phyre.get_default_100k_cache('ball')
        cache_actions = cache.action_array

        self.to_train()
        actions = np.zeros((len(tasks), 3))
        pipelines = []
        num_batches = 1 + len(tasks) // 64

        if self.dijkstra:
            all_distance_maps = T.zeros(len(tasks), 1, self.width, self.width)
            for task_idx in range(len(tasks)):
                dm_init_scene = sim.initial_scenes[task_idx]
                img = cv2.resize(phyre.observations_to_float_rgb(dm_init_scene), (self.width, self.width),
                                 cv2.INTER_MAX)  # read image
                target = np.logical_or(all_initial_scenes[task_idx, 1] == 1, all_initial_scenes[task_idx, 2] == 1)
                # cv2.imwrite('maze-initial.png', img)
                distance_map = find_distance_map_obj(img, target)
                all_distance_maps[task_idx, 0] = T.from_numpy(distance_map)

        for batch in range(num_batches):
            init_scenes = all_initial_scenes[batch * 64:64 * (batch + 1)].to(self.device)
            if self.dijkstra:
                distance_maps = all_distance_maps[batch * 64:64 * (batch + 1)].to(self.device)
            # emb_init_scenes = F.embedding(T.tensor(batch_scenes), T.eye(phyre.NUM_COLORS)).transpose(-1,-3)[:,1:6].float()
            # print(init_scenes.equal(emb_init_scenes))

            # FORWARD PASS
            with T.no_grad():
                base_paths = self.models["base_net"](init_scenes)
                if self.dijkstra:
                    target_paths = self.models["tar_net"](T.cat((init_scenes, distance_maps), dim=1))
                else:
                    target_paths = self.models["tar_net"](init_scenes)
                action_paths = self.models["act_net"](T.cat((init_scenes, target_paths, base_paths), dim=1))
                action_balls = self.models["ext_net"](T.cat((init_scenes, target_paths, action_paths), dim=1))
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, action_balls), dim=1)
                # text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred']
                # vis_batch(print_batch, f'result/flownet/solver/{self.path}', f'{batch}', text=text)

            batch_tasks = tasks[64 * batch:64 * (batch + 1)]
            os.makedirs(f'result/solver/generative/', exist_ok=True)
            for idx, ball in enumerate(action_balls[:, 0].cpu()):
                tried_actions = []
                task = batch_tasks[idx]

                # CHOOSE ONE VECTOR EXTRACTION METHOD
                # a = pic_to_action_vector(ball, r_fac=1.5)
                mask = np.max(init_scenes[idx].cpu().numpy(), axis=0)
                # print(mask.shape)
                a = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                       updates=5)
                a2 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a3 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a4 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a5 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                # print(a)

                drawn = draw_ball(self.width, a[0], a[1], a[2], invert_y=True).to(self.device)
                drawn2 = draw_ball(self.width, a2[0], a2[1], a2[2], invert_y=True).to(self.device)
                drawn3 = draw_ball(self.width, a3[0], a3[1], a3[2], invert_y=True).to(self.device)
                drawn4 = draw_ball(self.width, a4[0], a4[1], a4[2], invert_y=True).to(self.device)
                drawn5 = draw_ball(self.width, a5[0], a5[1], a5[2], invert_y=True).to(self.device)

                def get_nice_xyr(a):
                    x, y, r = str(round(a[0], 2)), str(round(a[1], 2)), str(round(a[2], 2))
                    return f"{x} {y} {r}"

                """
                pure_scene = T.as_tensor(np.max(print_batch[idx,[0,1,2,3,4]].numpy(), axis=0))
                scene_with_estimate = T.as_tensor(np.max(print_batch[idx,[0,1,2,3,4,-1]].numpy(), axis=0))
                scene_with_injected = T.as_tensor(np.max(T.stack((scene_with_estimate, drawn), dim=0).numpy(), axis=0))
                init_with_injected = T.as_tensor(np.max(T.cat((init_scenes[idx], drawn[None]), dim=0).numpy(), axis=0))
                pipeline = T.cat((print_batch[idx], scene_with_estimate[None], pure_scene[None], scene_with_injected[None], init_with_injected[None]), dim=0)

                text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'full\nscene', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred','action\nestimate', 'estimate\n/w action','   final\n   action']
                x,y,r = round(a[0], 3), round(a[1], 3), round(a[2], 3)
                vis_batch(pipeline[None], f'result/flownet/solving/generative', f"{task}__{str(a)}", text = text)
                """

                back = init_scenes[None, idx, 3:].sum(dim=1)[:, None]
                back = back / max(back.max(), 1)
                inits = init_scenes[None, idx, None]
                vis_line = T.cat((
                    T.stack((back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),
                    # inital scene
                    T.stack((back, base_paths[idx, None] + back, inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),
                    # scene with base
                    T.stack((action_paths[idx, None] + back, inits[:, :, 0] + target_paths[idx, None] + back,
                             inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),  # scene with action path and target
                    T.stack(
                        (action_balls[idx, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                        dim=-1),  # scene with action ball
                    T.stack((drawn[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn2[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn3[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn4[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn5[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1)),
                    dim=1).detach()

                vis_line = self.cut_off(vis_line.cpu())
                white = T.ones_like(vis_line)
                white[:, :, :, :, [0, 1]] -= vis_line[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [0, 2]] -= vis_line[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [1, 2]] -= vis_line[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                vis_line = self.cut_off(white)
                text = ['initial\nscene', 'distance_map', 'scene with\nbase path\nprediction',
                        'action and\ntarget path\nprediction', 'scene with\naction ball\nprediction',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a2)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a3)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a4)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a5)}']
                if self.dijkstra:
                    dm = distance_maps[None, idx]
                    print_dms = T.stack((dm, dm, dm), dim=-1)
                    vis_line = T.cat((vis_line, print_dms), dim=1)
                    text.append("dijkstra")
                vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}', f"{task}", text=text,
                          save=True, font_size=9)
                vis_line = vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}', f"{task}",
                                     text=text, save=False, font_size=9)

                # pipelines.append(vis_line)

                # Radius times 4 since actions are scaled this way (and times 2 to get diameter??)
                a[2] = a[2] * 4 * 2
                a2[2] = a2[2] * 4 * 2
                a3[2] = a3[2] * 4 * 2
                a4[2] = a4[2] * 4 * 2
                a5[2] = a5[2] * 4 * 2
                # print(a)
                # saving action
                base_actions = [a, a2, a3, a4, a5]
                actions[idx + batch * 64] = a

                # SIMULATING ACTION
                # vis setup
                vis_count = 0
                vis_max_count = 30 - 1
                vis_wid = 64
                vis_text_actions = []
                gif_stack = T.zeros(vis_max_count + 2, 10, vis_wid, vis_wid, 3)

                # setup for simulation
                action = random.choice(base_actions)
                delta_generator = action_delta_generator(pure_noise=pure_noise)
                task_idx = tasks.index(task)

                # First try:
                res = sim.simulate_action(task_idx, action)
                eva.maybe_log_attempt(task_idx, res.status)
                tried_actions.append(action)

                t = 0
                warning_flag = False
                while True:

                    # GIFIFY if VALID attempt
                    if not res.status.is_invalid() and vis_count < vis_max_count:
                        for i in range(min(len(res.images), 10)):
                            gif_stack[vis_count, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        vis_text_actions.append(str(np.round(action, decimals=2)))
                        vis_count += 1

                    # Check if SOLVED
                    if res.status.is_solved():

                        # GIFIFY
                        for i in range(min(len(res.images), 10)):
                            gif_stack[vis_max_count, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        while len(vis_text_actions) < vis_max_count:
                            vis_text_actions.append('')
                        vis_text_actions.append(
                            str(np.round(action, decimals=2)) + f"\ntry {eva.attempts_per_task_index[task_idx]}")

                        print()
                        print(f"{task} solved after", eva.attempts_per_task_index[task_idx])
                        # loop untill 100 actions:
                        while eva.attempts_per_task_index[task_idx] < 100:
                            eva.maybe_log_attempt(task_idx, res.status)

                    else:
                        # Try NEXT ATTEMPT:
                        if t < 2000:
                            action = random.choice(base_actions)
                            while self.similar_action_tried(action, tried_actions):
                                delta = delta_generator.__next__()
                                action = action + delta
                                # print("t", t, "delta:", delta, end='\n')

                            res = sim.simulate_action(task_idx, action, need_featurized_objects=False)
                            eva.maybe_log_attempt(task_idx, res.status)
                            tried_actions.append(action)
                            t += 1
                        else:
                            if not warning_flag:
                                print()
                                print(f"WARNING can't find valid action for {task}")
                            warning_flag = True
                            error = True
                            eva.maybe_log_attempt(task_idx, phyre.SimulationStatus.NOT_SOLVED)

                    # check if enough attempts
                    if eva.attempts_per_task_index[task_idx] >= 100:
                        # GIFIFY the gif_stack
                        if not res.status.is_solved():
                            vis_text_actions.append('')
                            print()
                            print(f"{task} not solved")

                        # ADD GT ROLLOUT
                        # getting GT action
                        GT_valid_flag = False
                        while not GT_valid_flag:
                            solving_actions = cache_actions[cache.load_simulation_states(task) == 1]
                            if len(solving_actions) == 0:
                                print("no solution action in cache at task", task)
                                solving_actions = [np.random.rand(3)]
                            action = random.choice(solving_actions)
                            # simulating GT action
                            res = sim.simulate_action(task_idx, action)
                            if not res.status.is_invalid():
                                GT_valid_flag = True
                            else:
                                print("invalid GT action", task, action)
                        for i in range(min(len(res.images), 10)):
                            gif_stack[-1, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        vis_text_actions.append(f"GT")

                        gifify(gif_stack, f'result/flownet/solving/{self.path}/{self.run}/{name}', f'{task}_tries',
                               text=vis_text_actions, constant=vis_line)
                        # break out and continue with next task:
                        break

        # print(list(zip(tasks,actions)))
        return eva.get_auccess()

    def combi_auccess(self, tasks, name, pure_noise=False, gt_paths=False, return_proposals=False):
        sim = phyre.initialize_simulator(tasks, 'ball')
        all_initial_scenes = T.tensor(
            [[cv2.resize((scene == channel).astype(float), (self.width, self.width)) for channel in range(2, 7)] for
             scene in sim.initial_scenes]).float().flip(-2)
        eva = phyre.Evaluator(tasks)
        cache = phyre.get_default_100k_cache('ball')
        cache_actions = cache.action_array

        self.to_train()
        actions = np.zeros((len(tasks), 3))
        pipelines = []
        num_batches = 1 + len(tasks) // 64
        # BACTHING ALL TASKS
        for batch in range(num_batches):
            init_scenes = all_initial_scenes[batch * 64:64 * (batch + 1)].to(self.device)
            # emb_init_scenes = F.embedding(T.tensor(batch_scenes), T.eye(phyre.NUM_COLORS)).transpose(-1,-3)[:,1:6].float()
            # print(init_scenes.equal(emb_init_scenes))

            # FORWARD PASS
            with T.no_grad():
                base_paths = self.models["base_net"](init_scenes)
                target_paths = self.models["tar_net"](init_scenes)
                action_paths = self.models["act_net"](T.cat((init_scenes, target_paths, base_paths), dim=1))
                action_balls = self.models["ext_net"](T.cat((init_scenes, target_paths, action_paths), dim=1))
                # print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths, action_balls), dim=1)
                # text = ['green\nball GT', 'blue GT\ndynamic', 'blue GT\nstatic', 'grey', 'black', 'base\npred', 'target\npred', 'action\npred', 'a-ball\npred']
                # vis_batch(print_batch, f'result/flownet/solver/{self.path}', f'{batch}', text=text)

            batch_tasks = tasks[64 * batch:64 * (batch + 1)]
            os.makedirs(f'result/solver/generative/', exist_ok=True)
            # LOOPING THROUGH ALL TASKS IN BATCH
            for idx, ball in enumerate(action_balls[:, 0].cpu()):
                task = batch_tasks[idx]

                # CHOOSE ONE VECTOR EXTRACTION METHOD
                # a = pic_to_action_vector(ball, r_fac=1.5)
                mask = np.max(init_scenes[idx].cpu().numpy(), axis=0)
                # print(mask.shape)
                a = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                       updates=5)
                a2 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a3 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a4 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a5 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=self.num_seeds, check_border=True, mask=mask,
                                        updates=5)
                a6 = grow_action_vector(ball, r_fac=self.r_fac, num_seeds=5, check_border=True, mask=mask, updates=5)
                # print(a)

                drawn = draw_ball(self.width, a[0], a[1], a[2], invert_y=True).to(self.device)
                drawn2 = draw_ball(self.width, a2[0], a2[1], a2[2], invert_y=True).to(self.device)
                drawn3 = draw_ball(self.width, a3[0], a3[1], a3[2], invert_y=True).to(self.device)
                drawn4 = draw_ball(self.width, a4[0], a4[1], a4[2], invert_y=True).to(self.device)
                drawn5 = draw_ball(self.width, a5[0], a5[1], a5[2], invert_y=True).to(self.device)
                drawn6 = draw_ball(self.width, a6[0], a6[1], a6[2], invert_y=True).to(self.device)

                def get_nice_xyr(a):
                    x, y, r = str(round(a[0], 2)), str(round(a[1], 2)), str(round(a[2], 2))
                    return f"{x} {y} {r}"

                back = init_scenes[None, idx, 3:].sum(dim=1)[:, None]
                back = back / max(back.max(), 1)
                inits = init_scenes[None, idx, None]
                vis_line = T.cat((
                    T.stack((back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),
                    # inital scene
                    T.stack((back, base_paths[idx, None] + back, inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),
                    # scene with base
                    T.stack((action_paths[idx, None] + back, inits[:, :, 0] + target_paths[idx, None] + back,
                             inits[:, :, 1] + inits[:, :, 2] + back), dim=-1),  # scene with action path and target
                    T.stack(
                        (action_balls[idx, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                        dim=-1),  # scene with action ball
                    T.stack((drawn[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn2[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn3[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn4[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1),
                    T.stack((drawn5[None, None] + back, inits[:, :, 0] + back, inits[:, :, 1] + inits[:, :, 2] + back),
                            dim=-1)),
                    dim=1).detach()
                vis_line = self.cut_off(vis_line.cpu())
                white = T.ones_like(vis_line)
                white[:, :, :, :, [0, 1]] -= vis_line[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [0, 2]] -= vis_line[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [1, 2]] -= vis_line[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                vis_line = self.cut_off(white)
                text = ['initial\nscene', 'scene with\nbase path\nprediction', 'action and\ntarget path\nprediction',
                        'scene with\naction ball\nprediction', f'injected x,y,r\naction vector\n{get_nice_xyr(a)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a2)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a3)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a4)}',
                        f'injected x,y,r\naction vector\n{get_nice_xyr(a5)}']
                vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f"seeds", text=text,
                          save=True, font_size=9)
                vis_line = vis_batch(vis_line, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f"seeds",
                                     text=text, save=False, font_size=9)

                # pipelines.append(vis_line)

                # Radius times 4 since actions are scaled this way (and times 2 to get diameter??)
                a[2] = a[2] * 4 * 2
                a2[2] = a2[2] * 4 * 2
                a3[2] = a3[2] * 4 * 2
                a4[2] = a4[2] * 4 * 2
                a5[2] = a5[2] * 4 * 2
                a6[2] = a6[2] * 4 * 2

                # print(a)
                # saving action
                base_actions = [a, a2, a3, a4, a5]
                actions[idx + batch * 64] = a
                delta_generator = action_delta_generator(pure_noise=pure_noise)

                # MAKE MORE ACTIONS
                tried_actions = []
                num_proposals = 100
                action_proposals = np.zeros((6, num_proposals, 3))
                ranked_actions = np.zeros((5 * num_proposals, 3))
                ranked_actions_viz = T.zeros(5 * num_proposals, 5, self.width, self.width, 3)
                for seed_id in range(5):
                    # print(task, "seed",seed_id)
                    for proposal_idx in range(num_proposals):
                        action = base_actions[seed_id]
                        tmp_base_action = action
                        while self.similar_action_tried(action, tried_actions) or self.is_invalid(action, mask):
                            delta = delta_generator.__next__()
                            action = np.clip(action + delta, 0, 1)
                        # print("delta", action-tmp_base_action)
                        tried_actions.append(action)
                        action_proposals[seed_id, proposal_idx] = action

                    if seed_id < 5:  # RANKING ACTIONS
                        # action_proposals[seed_id,:,2]
                        seed_ranked_actions, seed_rank_viz = self.rank_actions(task, init_scenes[idx],
                                                                               action_proposals[seed_id],
                                                                               seed_id=seed_id, name=name,
                                                                               gt_paths=gt_paths)
                        ranked_actions[seed_id::5] = seed_ranked_actions
                        # print(ranked_actions_viz.shape, seed_rank_viz.shape)
                        ranked_actions_viz[seed_id::5] = seed_rank_viz

                task_idx = tasks.index(task)

                # SIMULATOR RESULTS FROM SINGLE
                gt_conf = T.zeros(20)
                gt_scene = T.zeros(20, 1, self.width, self.width, 3)
                gt_confpic = T.ones(20, 1, self.width, self.width, 3)
                # get GT "confidence"
                for loc_idx, local_action in enumerate(np.moveaxis(action_proposals[:5], 0, 1).reshape(-1, 3)[:20]):
                    res = sim.simulate_action(task_idx, local_action)
                    # print(res.status.is_solved(), res.status.is_invalid())
                    if not res.status.is_invalid():
                        rollout = np.mean(np.array([phyre.observations_to_uint8_rgb(img) for img in res.images]),
                                          axis=0)
                        gt_scene[loc_idx, 0] = T.from_numpy(cv2.resize(rollout, (self.width, self.width))).float() / 255
                    if res.status.is_solved():
                        # print(res.status.is_solved(), float(res.status.is_solved()))
                        gt_conf[loc_idx] = float(res.status.is_solved())
                    else:
                        gt_conf[loc_idx] = 0.5 * float(res.status.is_invalid())
                    gt_confpic[loc_idx] *= gt_conf[loc_idx]
                # print(gt_conf)

                rank_viz = ranked_actions_viz.view(num_proposals, 5 * 5, self.width, self.width, 3)[:20]
                rank_viz = T.cat((rank_viz, gt_scene, gt_confpic), dim=1)
                text = ["initial\nproposal", "predicted\nscene", "predicted\nconf", "gt conf",
                        "simulator\nscene"] * 5 + ["first stage\nproposal", "gt conf"]
                vis_batch(rank_viz, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f'best-5-ranking',
                          text=text)

                # SIMULATING ACTIONS
                # vis setup
                vis_count = 0
                vis_max_count = 9
                vis_wid = 64
                vis_text_actions = []
                gif_stack = T.zeros(vis_max_count + 2, 10, vis_wid, vis_wid, 3)

                # setup for simulation

                # First try:
                tried_actions = []
                t = 0
                action = ranked_actions[t]
                res = sim.simulate_action(task_idx, action)
                eva.maybe_log_attempt(task_idx, res.status)
                tried_actions.append(action)
                t += 1

                warning_flag = False
                while True:

                    # GIFIFY if VALID attempt
                    if not res.status.is_invalid() and vis_count < vis_max_count:
                        for i in range(min(len(res.images), 10)):
                            gif_stack[vis_count, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        vis_text_actions.append(str(np.round(action, decimals=2)))
                        vis_count += 1

                    # Check if SOLVED
                    if res.status.is_solved():

                        # GIFIFY
                        for i in range(min(len(res.images), 10)):
                            gif_stack[vis_max_count, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        while len(vis_text_actions) < vis_max_count:
                            vis_text_actions.append('')
                        vis_text_actions.append(
                            str(np.round(action, decimals=2)) + f"\ntry {eva.attempts_per_task_index[task_idx]}")

                        print()
                        print(f"{task} solved after", eva.attempts_per_task_index[task_idx], "with",
                              ranked_actions[t - 1])
                        # loop untill 100 actions:
                        while eva.attempts_per_task_index[task_idx] < 100:
                            eva.maybe_log_attempt(task_idx, res.status)

                    # Try NEXT ATTEMPT:
                    else:
                        if t < num_proposals * 5:
                            action = ranked_actions[t]
                            while self.similar_action_tried(action, tried_actions):
                                delta = delta_generator.__next__()
                                action = action + delta
                                print("t", t, "delta:", delta, end='\n')

                            res = sim.simulate_action(task_idx, action, need_featurized_objects=False)
                            eva.maybe_log_attempt(task_idx, res.status)
                            tried_actions.append(action)
                            t += 1
                        else:
                            if not warning_flag:
                                print()
                                print(f"WARNING can't find valid action for {task}")
                            warning_flag = True
                            error = True
                            eva.maybe_log_attempt(task_idx, phyre.SimulationStatus.NOT_SOLVED)

                    # check if enough attempts
                    if eva.attempts_per_task_index[task_idx] >= 100:
                        # GIFIFY the gif_stack
                        if not res.status.is_solved():
                            vis_text_actions.append('')
                            print()
                            print(f"{task} not solved")

                        # ADD GT ROLLOUT
                        # getting GT action
                        GT_valid_flag = False
                        while not GT_valid_flag:
                            solving_actions = cache_actions[cache.load_simulation_states(task) == 1]
                            if len(solving_actions) == 0:
                                print("no solution action in cache at task", task)
                                solving_actions = [np.random.rand(3)]
                            action = random.choice(solving_actions)
                            # simulating GT action
                            res = sim.simulate_action(task_idx, action)
                            if not res.status.is_invalid():
                                GT_valid_flag = True
                            else:
                                print("invalid GT action", task, action)
                        for i in range(min(len(res.images), 10)):
                            gif_stack[-1, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        vis_text_actions.append(f"GT")

                        gifify(gif_stack, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f'tries',
                               text=vis_text_actions, constant=vis_line)
                        # break out and continue with next task:
                        break

        # print(list(zip(tasks,actions)))
        return eva.get_auccess()

    def rank_actions(self, task, init_scene, actions, seed_id=0, name="default", gt_paths=False):
        bs = 100
        repeated_init_scene = init_scene.repeat(bs, 1, 1, 1)
        all_confs = T.zeros(len(actions))
        sim = phyre.initialize_simulator([task], 'ball')

        drawn_actions = T.zeros(len(actions), 1, self.width, self.width)
        for a_idx, a in enumerate(actions):  # draw all actions
            drawn_actions[a_idx] = draw_ball(self.width, a[0], a[1], 0.125 * a[2], invert_y=True)

        # SIMULATOR RESULTS
        gt_conf = T.zeros(bs)
        gt_scene = T.zeros(bs, 1, self.width, self.width, 3)
        gt_conf_pic = T.ones(bs, 1, self.width, self.width, 3)
        if gt_paths:
            red = T.zeros(bs, 1, self.width, self.width)
            green = T.zeros(bs, 1, self.width, self.width)
            blue = T.zeros(bs, 1, self.width, self.width)
        # get GT "confidence"
        for loc_idx, local_action in enumerate(actions):
            res = sim.simulate_action(0, local_action, stride=5)
            # print(res.status.is_solved(), res.status.is_invalid())
            if not res.status.is_invalid():
                rollout = np.mean(np.array([phyre.observations_to_uint8_rgb(img) for img in res.images]), axis=0)
                if gt_paths:
                    red[loc_idx, 0] = T.from_numpy(np.max(np.array(
                        [cv2.resize(np.flip(img == 1, axis=0).astype(float), (self.width, self.width)) for img in
                         res.images]), axis=0))
                    green[loc_idx, 0] = T.from_numpy(np.max(np.array(
                        [cv2.resize(np.flip(img == 2, axis=0).astype(float), (self.width, self.width)) for img in
                         res.images]), axis=0))
                    blue[loc_idx, 0] = T.from_numpy(np.max(np.array(
                        [cv2.resize(np.flip(img == 3, axis=0).astype(float), (self.width, self.width)) for img in
                         res.images]), axis=0))
                gt_scene[loc_idx, 0] = T.from_numpy(cv2.resize(rollout, (self.width, self.width))).float() / 255
            if res.status.is_solved():
                # print(res.status.is_solved(), float(res.status.is_solved()))
                gt_conf[loc_idx] = float(res.status.is_solved())
            else:
                gt_conf[loc_idx] = 0.5 * float(res.status.is_invalid())
            gt_conf_pic[loc_idx] *= gt_conf[loc_idx]

        action_batch = drawn_actions.to(self.device)
        scene_batch = repeated_init_scene[:action_batch.shape[0]]
        with T.no_grad():
            if not gt_paths:
                red = self.models["sim_net"](T.cat((action_batch, scene_batch), dim=1))
                green = self.models["sim_net"](
                    T.cat((scene_batch[:, None, 0], action_batch, scene_batch[:, 1:]), dim=1))
                blue = self.models["sim_net"](
                    T.cat((scene_batch[:, None, 1], action_batch, scene_batch[:, [0, 2, 3, 4]]), dim=1))
            else:
                red = red.to(self.device)
                green = green.to(self.device)
                blue = blue.to(self.device)
            conf = self.models["success_net"](T.cat((action_batch, scene_batch, red, green, blue), dim=1))

        # RANKING
        conf_order = T.argsort(conf[:, 0], descending=True)
        actions = T.tensor(actions)
        ranked_actions = actions[conf_order]

        # VISUALIZATION
        conf_pic = conf[:, :, None, None] * T.ones_like(red)
        # print(red.shape, gt_conf.shape, gt_conf[:,None,None,None].shape)
        background = scene_batch[:, 3:].sum(dim=1)[:, None]
        background = background / max(background.max(), 1)
        tmp_conf = conf_pic
        scene = T.stack((action_batch + background, scene_batch[:, None, 0] + background,
                         scene_batch[:, None, 1] + scene_batch[:, None, 2] + background), dim=-1)
        diff_batch = T.cat((
            scene,
            scene + T.stack((red, green, blue), dim=-1),
            0.5 * T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1)),
            dim=1).detach()
        diff_batch = self.cut_off(diff_batch.cpu())
        white = T.ones_like(diff_batch)
        white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
        white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
        white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
        diff_batch = self.cut_off(white)
        diff_batch = T.cat((diff_batch, gt_scene, gt_conf_pic), dim=1)
        diff_batch = diff_batch[conf_order]
        text = ["initial\nproposal", "predicted\nscene", "predicted\nconf", "simulated\ninitial\nscene", "gt conf"]
        descr = [str(v.item()) for v in conf[conf_order, 0]]
        vis_batch(diff_batch, f'result/flownet/solving/{self.path}/{self.run}/{name}/{task}', f'ranking_{seed_id}',
                  text=text, rows=[str(rowidx) for rowidx in range(bs)], descr=descr)
        conf.detach()[:, 0]

        return ranked_actions.numpy(), diff_batch

    def generative_auccess_old(self, eval_setup, fold, train_mode='CONS', epochs=1):
        self.to_train()
        data_loader = self.test_dataloader
        tar_net = self.models["tar_net"]
        base_net = self.models["base_net"]
        act_net = self.models["act_net"]
        ext_net = self.models["ext_net"]

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)
                # Prepare Data
                action_balls = X[:, 0]
                init_scenes = X[:, 1:6]
                base_paths = X[:, 6]
                target_paths = X[:, 7]
                goal_paths = X[:, 8]
                action_paths = X[:, 9]

                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                # FORWARD PASS
                with T.no_grad():
                    if train_mode == 'MIX':
                        modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                    else:
                        modus = train_mode

                    # Forward Pass
                    target_pred = tar_net(init_scenes)
                    base_pred = base_net(init_scenes)
                    if modus == 'GT':
                        action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
                    elif modus == 'CONS':
                        action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
                    elif modus == 'COMB':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                    elif modus == 'END':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

                # VISUALIZATION
                os.makedirs(f'result/flownet/inspect/{self.path}', exist_ok=True)
                Z = X.cpu()
                vis_batch(T.stack((Z, T.zeros_like(Z), T.zeros_like(Z)), dim=-1), "result/test", "color",
                          text=["hello!"])

                print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action',
                        'base', 'target', 'action', 'red ball']
                vis_batch(print_batch.cpu(), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}',
                          f'poch_{epoch}_{i}', text=text)

                # sum_batch = T.cat((base_paths[:,None]+base_pred, target_paths[:,None]+target_pred, action_paths[:,None]+action_pred, action_balls[:,None]+ball_pred), dim=1).detach().abs()/2
                # text = ['base\npaths', 'target\npaths', 'action\npaths', 'action\nballs']
                # vis_batch(sum_batch.cpu(), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_sum', text=text)

                # Extract action and draw:
                drawings = T.zeros_like(ball_pred)
                print("drawing balls for batch", i)
                for b_idx, ball in enumerate(ball_pred[:, 0].cpu()):
                    a = grow_action_vector(ball, r_fac=self.r_fac, check_border=True)
                    # print(a)
                    drawn = draw_ball(self.width, *a, invert_y=True)
                    drawings[b_idx, 0] = drawn

                background = init_scenes[:, 3:].sum(dim=1)[:, None]
                background = background / max(background.max(), 1)
                diff_batch = T.cat((
                    T.stack((background, init_scenes[:, None, 0] + background,
                             init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                    T.stack((base_pred, base_paths[:, None], T.zeros_like(base_pred)), dim=-1),
                    T.stack((target_pred, target_paths[:, None], T.zeros_like(target_pred)), dim=-1),
                    T.stack((action_pred, action_paths[:, None], T.zeros_like(action_pred)), dim=-1),
                    T.stack((ball_pred, action_balls[:, None], T.zeros_like(ball_pred)), dim=-1),
                    T.stack((ball_pred + background, init_scenes[:, None, 0] + background,
                             init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                    T.stack((drawings + background, init_scenes[:, None, 0] + background,
                             init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                    T.stack((action_balls[:, None] + background, init_scenes[:, None, 0] + background,
                             init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)),
                    dim=1).detach()
                diff_batch = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                diff_batch = white
                text = ['initial\nscene', 'base\npaths', 'target\npaths', 'action\npaths', 'action\nballs',
                        'overlay\nscene', 'injected\n scene', 'GT\nscene']
                vis_batch(self.cut_off(diff_batch.cpu()),
                          f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_diff',
                          text=text)

    def brute_searched_actions(self, tasks, all_initial_scenes):
        cache = self.cache
        n_actions = 1000
        actions = cache.action_array[:n_actions]
        solutions = np.zeros((len(tasks), 100, 3))

        self.to_train()
        all_initial_scenes = all_initial_scenes.to(self.device)

        # pre-compute bases
        with T.no_grad():
            # all_base_paths = self.models["base_net"](init_scenes)
            pass

        # loop through tasks
        for j, task in enumerate(tasks):
            # loop batched through all actions
            confs = T.zeros(n_actions)
            bs = 64
            n_batches = 1 + n_actions // bs
            # repeated_base_paths = all_base_paths[j].repeat(bs,1,1,1,1)
            repeated_initial_scenes = all_initial_scenes[j].repeat(bs, 1, 1, 1)
            for i in range(n_batches):
                print(f"at action {i * bs} for task {task}", end='\r')
                action_batch = actions[i * bs:(i + 1) * bs]
                init_scenes = repeated_initial_scenes[:len(action_batch)]
                # base_paths = repeated_base_paths[:len(action_batch)]

                # generate action pics from vectors
                action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
                for k, action_vector in enumerate(action_batch):
                    action_pics[k, 0] = draw_ball(self.width, *action_vector, invert_y=True)
                action_pics = action_pics.to(self.device)

                with T.no_grad():
                    base_paths = self.models["base_net"](T.cat((action_pics, init_scenes), dim=1))
                    action_paths = self.models["act_net"](T.cat((action_pics, init_scenes, base_paths), dim=1))
                    target_paths = self.models["tar_net"](
                        T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                    confidence = self.models["ext_net"](
                        T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))

                confs[i * bs:(i + 1) * bs] = confidence[:, 0]

            conf_args = T.argsort(confs, descending=True)
            # print(conf_args[:100])
            # print(actions[conf_args[:100]])
            best_actions = actions[conf_args[:100]]
            solutions[j] = best_actions

        return solutions

    def brute_auccess(self, tasks):
        cache = phyre.get_default_100k_cache('ball')
        n_actions = 1000
        actions = cache.action_array[:n_actions]
        n_try = 500
        solutions = np.zeros((len(tasks), n_try, 3))

        sim = phyre.initialize_simulator(tasks, 'ball')
        all_initial_scenes = T.tensor(
            [[cv2.resize((scene == channel).astype(float), (self.width, self.width)) for channel in range(2, 7)] for
             scene in sim.initial_scenes]).float().flip(-2)
        eva = phyre.Evaluator(tasks)
        # vis_batch(all_initial_scenes, "result/test", "initial_scenes_vis")

        self.to_train()
        all_initial_scenes = all_initial_scenes.to(self.device)

        # loop through tasks
        for j, task in enumerate(tasks):
            # COLLECT 100 BEST: loop batched through all potential actions
            confs = T.zeros(n_actions)
            bs = 64
            n_batches = 1 + n_actions // bs
            # repeated_base_paths = all_base_paths[j].repeat(bs,1,1,1,1)
            repeated_initial_scenes = all_initial_scenes[j].repeat(bs, 1, 1, 1)
            for i in range(n_batches):
                print(f"at action {i * bs} for task {task}", end='\r')
                action_batch = actions[i * bs:(i + 1) * bs]
                init_scenes = repeated_initial_scenes[:len(action_batch)]
                # base_paths = repeated_base_paths[:len(action_batch)]

                # generate action pics from vectors
                action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
                for k, action_vector in enumerate(action_batch):
                    x, y, r = action_vector
                    action_pics[k, 0] = draw_ball(self.width, x, y, r * 0.125, invert_y=True)
                action_pics = action_pics.to(self.device)

                with T.no_grad():
                    base_paths = self.models["base_net"](T.cat((action_pics, init_scenes), dim=1))
                    action_paths = self.models["act_net"](T.cat((action_pics, init_scenes, base_paths), dim=1))
                    target_paths = self.models["tar_net"](
                        T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                    confidence = self.models["ext_net"](
                        T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))
                confs[i * bs:(i + 1) * bs] = confidence[:, 0]

            # SORT OR SIMPLY TAKE GOOD ONES?
            conf_args = T.argsort(confs, descending=True)
            sorted_conf = T.sort(confs, descending=True)
            # conf_args = T.nonzero(confs>0.9).flatten()
            # sorted_conf = confs[conf_args]
            print("CONF ARGS", conf_args[:50])
            print("CONF", sorted_conf[:50])
            print("ACTIONS:", actions[conf_args[:50]])
            print("CACHE RESULTS:", cache.load_simulation_states(task)[conf_args[:50]])
            best_actions = actions[conf_args[:n_try]]
            solutions[j, :len(best_actions)] = best_actions

            # VISUALIZE best actions:
            # print(f"visualizing best actions for task {task}", end='\r')
            action_batch = best_actions
            repeated_initial_scenes = all_initial_scenes[j].repeat(len(action_batch), 1, 1, 1)
            init_scenes = repeated_initial_scenes
            # base_paths = repeated_base_paths[:len(action_batch)]
            # generate action pics from vectors
            action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
            for k, action_vector in enumerate(action_batch):
                x, y, r = action_vector
                action_pics[k, 0] = draw_ball(self.width, x, y, r * 0.125, invert_y=True)
            action_pics = action_pics.to(self.device)

            with T.no_grad():
                base_paths = self.models["base_net"](T.cat((action_pics, init_scenes), dim=1))
                action_paths = self.models["act_net"](T.cat((action_pics, init_scenes, base_paths), dim=1))
                target_paths = self.models["tar_net"](
                    T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                # init_scenes = T.cat((T.ones_like(init_scenes)[:10], init_scenes[10:]), dim=0)
                confidence = self.models["ext_net"](
                    T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))

                os.makedirs(f'result/flownet/solver/{self.path}/{self.run}', exist_ok=True)
                # print("conf",confidence[:20])
                print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths,
                                     T.ones_like(base_paths) * confidence[:, :, None, None]), dim=1).detach()
                text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\npath pred',
                        'target\npath pred', 'action\npath pred', 'conf']
                vis_batch(print_batch.cpu(), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions',
                          text=text)

                background = init_scenes[:, 4:].sum(dim=1)[:, None]
                background = background / max(background.max(), 1)
                tmp_conf = T.ones_like(base_paths) * confidence[:, :, None, None]
                scene = T.stack((action_pics + background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)
                green_target = T.stack((T.zeros_like(target_paths), target_paths, T.zeros_like(target_paths)), dim=-1)
                red_target = T.stack((action_paths, T.zeros_like(action_paths), T.zeros_like(action_paths)), dim=-1)
                diff_batch = T.cat((
                    scene,
                    scene + green_target + red_target,
                    T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1)),
                    dim=1).detach()
                diff_batch = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                diff_batch = white
                text = ['scene', 'scene\nprediction', 'confidence']
                vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/solving/{self.path}/{self.run}',
                          f'{task}_best_actions_diff', text=text)

            # EVALUATION
            vis_count = 0
            vis_max_count = 5
            vis_wid = 64
            vis_text_actions = []
            constant = vis_batch(self.cut_off(diff_batch.cpu()[:vis_max_count]),
                                 f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions_diff',
                                 text=text, save=False)
            gif_stack = T.zeros(vis_max_count + 1, 10, vis_wid, vis_wid, 3)
            for action in best_actions:
                if eva.attempts_per_task_index[j] >= 100:
                    # GIFIFY
                    if not res.status.is_solved():
                        vis_text_actions.append('')
                        print()
                        print(f"{task} not solved")
                    gifify(gif_stack, f'result/flownet/solving/{self.path}/{self.run}', f'{task}_tries_gif',
                           text=vis_text_actions, constant=constant)
                    break
                res = sim.simulate_action(j, action)
                eva.maybe_log_attempt(j, res.status)

                # GIFIFY
                if not res.status.is_invalid() and vis_count < vis_max_count:
                    for i in range(min(len(res.images), 10)):
                        gif_stack[vis_count, i] = T.tensor(
                            cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                    vis_text_actions.append(str(np.round(action, decimals=2)))
                    vis_count += 1

                if res.status.is_solved():
                    # GIFIFY
                    for i in range(min(len(res.images), 10)):
                        gif_stack[5, i] = T.tensor(
                            cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                    while len(vis_text_actions) < vis_max_count:
                        vis_text_actions.append('')
                    vis_text_actions.append(
                        str(np.round(action, decimals=2)) + f"\ntry {eva.attempts_per_task_index[j]}")

                    print()
                    print(f"{task} solved after", eva.attempts_per_task_index[j])
                    while eva.attempts_per_task_index[j] < 100:
                        eva.maybe_log_attempt(j, res.status)

        print("AUCCES", eva.get_auccess())
        return eva.get_auccess()

    def brute_proposals(self, initial_scene, actions):
        solutions = np.zeros((len(tasks), 500, 3))
        # vis_batch(all_initial_scenes, "result/test", "initial_scenes_vis")

        # loop through tasks
        # COLLECT 100 BEST: loop batched through all potential actions
        confs = T.zeros(n_actions)
        bs = 64
        n_batches = 1 + n_actions // bs
        # repeated_base_paths = all_base_paths[j].repeat(bs,1,1,1,1)
        repeated_initial_scenes = initiakl_scene.repeat(bs, 1, 1, 1)
        for i in range(n_batches):
            print(f"at action {i * bs} for task {task}", end='\r')
            action_batch = actions[i * bs:(i + 1) * bs]
            init_scenes = repeated_initial_scenes[:len(action_batch)]
            # base_paths = repeated_base_paths[:len(action_batch)]

            # generate action pics from vectors
            action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
            for k, action_vector in enumerate(action_batch):
                x, y, r = action_vector
                action_pics[k, 0] = draw_ball(self.width, x, y, r * 0.125, invert_y=True)
            action_pics = action_pics.to(self.device)

            with T.no_grad():
                base_paths = self.models["base_net"](T.cat((action_pics, init_scenes), dim=1))
                action_paths = self.models["act_net"](T.cat((action_pics, init_scenes, base_paths), dim=1))
                target_paths = self.models["tar_net"](
                    T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
                confidence = self.models["ext_net"](
                    T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))
            confs[i * bs:(i + 1) * bs] = confidence[:, 0]

        conf_args = T.argsort(confs, descending=True)
        sorted_conf = T.sort(confs, descending=True)
        print("CONF ARGS", conf_args[:50])
        print("CONF", sorted_conf[:50])
        print("ACTIONS:", actions[conf_args[:50]])
        print("CACHE RESULTS:", cache.load_simulation_states(task)[conf_args[:50]])
        best_actions = actions[conf_args[:500]]
        solutions[j] = best_actions

        # VISUALIZE best actions:
        # print(f"visualizing best actions for task {task}", end='\r')
        action_batch = best_actions
        repeated_initial_scenes = all_initial_scenes[j].repeat(len(action_batch), 1, 1, 1)
        init_scenes = repeated_initial_scenes
        # base_paths = repeated_base_paths[:len(action_batch)]
        # generate action pics from vectors
        action_pics = T.zeros(len(action_batch), 1, self.width, self.width)
        for k, action_vector in enumerate(action_batch):
            x, y, r = action_vector
            action_pics[k, 0] = draw_ball(self.width, x, y, r * 0.125, invert_y=True)
        action_pics = action_pics.to(self.device)

        with T.no_grad():
            base_paths = self.models["base_net"](T.cat((action_pics, init_scenes), dim=1))
            action_paths = self.models["act_net"](T.cat((action_pics, init_scenes, base_paths), dim=1))
            target_paths = self.models["tar_net"](T.cat((action_pics, init_scenes, base_paths, action_paths), dim=1))
            # init_scenes = T.cat((T.ones_like(init_scenes)[:10], init_scenes[10:]), dim=0)
            confidence = self.models["ext_net"](
                T.cat((action_pics, init_scenes, base_paths, action_paths, target_paths), dim=1))

            os.makedirs(f'result/flownet/solver/{self.path}/{self.run}', exist_ok=True)
            # print("conf",confidence[:20])
            print_batch = T.cat((init_scenes, base_paths, target_paths, action_paths,
                                 T.ones_like(base_paths) * confidence[:, :, None, None]), dim=1).detach()
            text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\npath pred',
                    'target\npath pred', 'action\npath pred', 'conf']
            vis_batch(print_batch.cpu(), f'result/flownet/solving/{self.path}/{self.run}', f'{task}_best_actions',
                      text=text)

            background = init_scenes[:, 4:].sum(dim=1)[:, None]
            background = background / max(background.max(), 1)
            tmp_conf = T.ones_like(base_paths) * confidence[:, :, None, None]
            scene = T.stack((action_pics + background, init_scenes[:, None, 0] + background,
                             init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)
            green_target = T.stack((T.zeros_like(target_paths), target_paths, T.zeros_like(target_paths)), dim=-1)
            red_target = T.stack((action_paths, T.zeros_like(action_paths), T.zeros_like(action_paths)), dim=-1)
            diff_batch = T.cat((
                scene,
                scene + green_target + red_target,
                T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1)),
                dim=1).detach()
            diff_batch = self.cut_off(diff_batch.cpu())
            white = T.ones_like(diff_batch)
            white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
            white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
            white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
            diff_batch = white
            text = ['scene', 'scene\nprediction', 'confidence']
            vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/solving/{self.path}/{self.run}',
                      f'{task}_best_actions_diff', text=text)

    def load_data(self, setup='ball_within_template', fold=0, train_tasks=[], test_tasks=[], brute_search=False,
                  n_per_task=1, shuffle=True, test=False, setup_name="all-tasks", proposal_dict=None,
                  sequential=False):
        fold_id = fold
        eval_setup = setup
        width = self.width
        batchsize = 32
        dijkstra_str = "_dijkstra" if self.dijkstra else ""

        if train_tasks and test_tasks:
            train_ids = train_tasks
            test_ids = test_tasks
        else:
            setup_name = "within" if setup == 'ball_within_template' else "cross"
            setup_name = setup_name + dijkstra_str
            if proposal_dict is not None:
                setup_name = setup_name + "_proposals"

            train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
            test_ids = dev_ids + test_ids

        if not test:
            if not sequential:
                make_mono_dataset_2(
                    f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n",
                    size=(width, width), tasks=train_ids[:], batch_size=batchsize // 2 if brute_search else batchsize,
                    n_per_task=n_per_task, shuffle=shuffle, proposal_dict=proposal_dict, dijkstra=self.dijkstra)
            else:
                make_mono_dataset_sequential(
                    f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n",
                    size=(width, width), tasks=train_ids[:], batch_size=batchsize // 2 if brute_search else batchsize,
                    n_per_task=n_per_task, shuffle=shuffle, proposal_dict=proposal_dict, dijkstra=self.dijkstra)
        else:
            self.test_dataloader, self.test_index = make_mono_dataset(
                f"data/{setup_name}_fold_{fold_id}_test_{width}xy_{n_per_task}n",
                size=(width, width), tasks=test_ids, n_per_task=n_per_task, shuffle=shuffle,
                proposal_dict=proposal_dict, dijkstra=self.dijkstra)
        if brute_search:
            if not test:
                self.failed_dataloader, self.failed_index = make_mono_dataset(
                    f"data/{setup_name}_fold_{fold_id}_failed_train_{width}xy_{n_per_task}n",
                    size=(width, width), tasks=train_ids[:], solving=False, batch_size=batchsize // 2,
                    n_per_task=n_per_task, shuffle=shuffle, proposal_dict=proposal_dict, dijkstra=self.dijkstra)
            else:
                self.failed_test_dataloader, self.failed_test_index = make_mono_dataset(
                    f"data/{setup_name}_fold_{fold_id}_failed_test_{width}xy_{n_per_task}n",
                    size=(width, width), tasks=test_ids[:], solving=False, batch_size=batchsize // 2,
                    n_per_task=n_per_task, shuffle=shuffle, proposal_dict=proposal_dict, dijkstra=self.dijkstra)
        os.makedirs(f'result/flownet/training/{self.path}', exist_ok=True)
        with open(f'result/flownet/training/{self.path}/namespace.txt', 'w') as handle:
            handle.write(f"{self.modeltype} {setup} {fold}")

    def make_visualisation(self, setup='ball_within_template', fold=0, train_tasks=[], test_tasks=[],
                           n_per_task=1, setup_name="all-tasks", proposal_dict=None,
                           model='sfm2'):

        if model == "sfm1":
            MODEL_PATH = f'./saves/flownet/{setup}/SfM1.pt'
            net = self.models["SfM1"]
            net.load_state_dict(T.load(MODEL_PATH))
            net.to(self.device).eval()
            save_img_dir = f'./proposalNet_result/{setup}/SfM1/grid'
        elif model == "sfm2":
            MODEL_PATH = f'./saves/flownet/{setup}/SfM2.pt'
            net = self.models["SfM2"]
            net.load_state_dict(T.load(MODEL_PATH))
            net.to(self.device).eval()
            save_img_dir = f'./proposalNet_result/{setup}/SfM2/grid'

        pic_id = 1

        fold_id = fold
        eval_setup = setup
        width = self.width

        if train_tasks and test_tasks:
            train_ids = train_tasks
            test_ids = test_tasks
        else:
            setup_name = "within" if setup == 'ball_within_template' else "cross"
            if proposal_dict is not None:
                setup_name = setup_name + "_proposals"

        with open("./phyre_get_fold.pickle", "rb") as f:
            x = pickle.load(f)
        train_ids, dev_ids, test_ids = x
        train_ids += dev_ids + test_ids
        train_ids = sorted(train_ids)

        # train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
        # train_ids += dev_ids + test_ids
        # train_ids = sorted(train_ids)

        data = make_mono_dataset_2(
            f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n",
            size=(width, width), tasks=train_ids, n_per_task=n_per_task,
            proposal_dict=proposal_dict, dijkstra=self.dijkstra, save=False)

        load_path = f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n"

        if model == "sfm1":
            obj_idxs = range(6)
            static_obj_idxs = [3, 5]
            rows = []
            for i in tqdm(range(len(data))):
                X = data[i]
                X = np.true_divide(X, 255.)
                init_scene = X[:6]
                paths = X[6:]

                row = []
                for obj_idx in obj_idxs:
                    # skip for static objects
                    if obj_idx in static_obj_idxs:
                        continue

                    # skip if object corresponding to obj_idx is not in scene
                    if not obj_in_scene(init_scene[obj_idx]):
                        continue

                    # Current object
                    current_obj_channel = init_scene[obj_idx]
                    current_obj_path_channel = paths[obj_idx]

                    # Dynamic objects
                    dynamic_obj_idxs = [idx for idx in obj_idxs if idx not in static_obj_idxs
                                        and obj_in_scene(init_scene[idx]) and idx != obj_idx]
                    dynamic_obj_channel = np.max(init_scene[dynamic_obj_idxs], axis=0)
                    dynamic_obj_path_channel = np.max(paths[dynamic_obj_idxs], axis=0)

                    # Static objects
                    static_obj_idxs_in_scene = [idx for idx in static_obj_idxs
                                                if obj_in_scene(init_scene[idx])]
                    if len(static_obj_idxs_in_scene) > 0:
                        static_obj_channel = np.max(init_scene[static_obj_idxs_in_scene], axis=0)
                    else:
                        static_obj_channel = np.zeros((init_scene.shape[1], init_scene.shape[2]))

                    current_obj_channel = np.expand_dims(current_obj_channel, axis=0)
                    dynamic_obj_channel = np.expand_dims(dynamic_obj_channel, axis=0)
                    static_obj_channel = np.expand_dims(static_obj_channel, axis=0)
                    current_obj_path_channel = np.expand_dims(current_obj_path_channel, axis=0)
                    dynamic_obj_path_channel = np.expand_dims(dynamic_obj_path_channel, axis=0)

                    # Processed data
                    input_initial_scene = np.concatenate([current_obj_channel, dynamic_obj_channel,
                                                          static_obj_channel], axis=0)

                    predicted_path = net(T.tensor(input_initial_scene).float().to(self.device)[None])[0][0]
                    predicted_path = predicted_path.cpu().detach().numpy()[:, :, None]

                    # visualisation
                    row.append(np.moveaxis(input_initial_scene, 0, -1))
                    row.append(np.concatenate([np.moveaxis(current_obj_path_channel, 0, -1),
                                               np.moveaxis(dynamic_obj_channel, 0, -1),
                                               predicted_path,
                                               np.moveaxis(static_obj_channel, 0, -1)], axis=-1))

                rows.append(row)
                if len(rows) == 100:
                    vis_pred_path_task(rows, save_img_dir, pic_id)
                    pic_id += 1
                    rows = []

        elif model == "sfm2":
            with gzip.open(load_path + '/sfm1_paths.pickle', 'rb') as fp:
                sfm1_paths = pickle.load(fp)

            obj_idxs = range(6)
            static_obj_idxs = [3, 5]
            rows = []
            for i in tqdm(range(len(data))):
                X = data[i]
                X = np.true_divide(X, 255.)
                init_scene = X[:6]
                paths = sfm1_paths[i]
                ground_truth_paths = X[6:]

                row = []
                for obj_idx in obj_idxs:
                    # skip for static objects
                    if obj_idx in static_obj_idxs:
                        continue

                    # skip if object corresponding to obj_idx is not in scene
                    if not obj_in_scene(init_scene[obj_idx]):
                        continue

                    # Current object
                    current_obj_channel = init_scene[obj_idx]
                    current_obj_path_channel = paths[obj_idx]

                    # Dynamic objects
                    dynamic_obj_idxs = [idx for idx in obj_idxs if idx not in static_obj_idxs
                                        and obj_in_scene(init_scene[idx]) and idx != obj_idx]
                    dynamic_obj_channel = np.max(init_scene[dynamic_obj_idxs], axis=0)
                    dynamic_obj_path_channel = np.max(paths[dynamic_obj_idxs], axis=0)

                    # Static objects
                    static_obj_idxs_in_scene = [idx for idx in static_obj_idxs
                                                if obj_in_scene(init_scene[idx])]
                    if len(static_obj_idxs_in_scene) > 0:
                        static_obj_channel = np.max(init_scene[static_obj_idxs_in_scene], axis=0)
                    else:
                        static_obj_channel = np.zeros((init_scene.shape[1], init_scene.shape[2]))

                    current_obj_channel = np.expand_dims(current_obj_channel, axis=0)
                    dynamic_obj_channel = np.expand_dims(dynamic_obj_channel, axis=0)
                    static_obj_channel = np.expand_dims(static_obj_channel, axis=0)
                    current_obj_path_channel = np.expand_dims(current_obj_path_channel, axis=0)
                    dynamic_obj_path_channel = np.expand_dims(dynamic_obj_path_channel, axis=0)

                    # Processed data
                    input_channels = np.concatenate([current_obj_channel, dynamic_obj_channel,
                                                     static_obj_channel, current_obj_path_channel,
                                                     dynamic_obj_path_channel], axis=0)

                    predicted_path = net(T.tensor(input_channels).float().to(self.device)[None])[0][0]
                    predicted_path = predicted_path.cpu().detach().numpy()[:, :, None]

                    # visualisation
                    row.append(np.moveaxis(input_channels[:3], 0, -1))  # appending the initial scene

                    row.append(np.concatenate([ground_truth_paths[obj_idx][:, :, None],
                                               np.moveaxis(dynamic_obj_channel, 0, -1),
                                               predicted_path,
                                               np.moveaxis(static_obj_channel, 0, -1)], axis=-1))

                rows.append(row)
                if len(rows) == 100:
                    vis_pred_path_task(rows, save_img_dir, pic_id)
                    pic_id += 1
                    rows = []

    def load_processed_data_sfm1(self, setup='ball_within_template', fold=0, train_tasks=[], test_tasks=[],
                                 brute_search=False,
                                 n_per_task=1, shuffle=True, test=False, setup_name="all-tasks", proposal_dict=None):
        fold_id = fold
        eval_setup = setup
        width = self.width
        batchsize = 32
        dijkstra_str = "_dijkstra" if self.dijkstra else ""

        if train_tasks and test_tasks:
            train_ids = train_tasks
            test_ids = test_tasks
        else:
            setup_name = "within" if setup == 'ball_within_template' else "cross"
            setup_name = setup_name + dijkstra_str
            if proposal_dict is not None:
                setup_name = setup_name + "_proposals"

        if not test:
            load_path = save_path = f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n"
            if os.path.exists(load_path + "/processed_data_sfm1.pickle"):
                with gzip.open(load_path + '/processed_data_sfm1.pickle', 'rb') as fp:
                    processed_data = pickle.load(fp)
                    X = T.tensor(processed_data).float()

                    with open(load_path + '/index.pickle', 'rb') as fp:
                        index = pickle.load(fp)

                    self.train_dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), batchsize,
                                                                    shuffle=False)
                    self.train_index = index

            else:
                if os.path.exists(load_path + "/data.pickle") and os.path.exists(load_path + "/index.pickle"):
                    with gzip.open(load_path + '/data.pickle', 'rb') as fp:
                        data = pickle.load(fp)
                        import random
                        random.seed(42)
                        random.shuffle(data)
                        procesed_data = []

                        obj_idxs = range(6)
                        static_obj_idxs = [3, 5]
                        for i in tqdm(range(len(data))):
                            X = data[i]
                            X = np.true_divide(X, 255.)
                            init_scene = X[:6]
                            paths = X[6:]

                            for obj_idx in obj_idxs:
                                # skip for static objects
                                if obj_idx in static_obj_idxs:
                                    continue

                                # skip if object corresponding to obj_idx is not in scene
                                if not obj_in_scene(init_scene[obj_idx]):
                                    continue

                                # Current object
                                current_obj_channel = init_scene[obj_idx]
                                current_obj_path_channel = paths[obj_idx]

                                # Dynamic objects
                                dynamic_obj_idxs = [idx for idx in obj_idxs if idx not in static_obj_idxs
                                                    and obj_in_scene(init_scene[idx]) and idx != obj_idx]
                                dynamic_obj_channel = np.max(init_scene[dynamic_obj_idxs], axis=0)

                                # Static objects
                                static_obj_idxs_in_scene = [idx for idx in static_obj_idxs
                                                            if obj_in_scene(init_scene[idx])]
                                if len(static_obj_idxs_in_scene) > 0:
                                    static_obj_channel = np.max(init_scene[static_obj_idxs_in_scene], axis=0)
                                else:
                                    static_obj_channel = np.zeros((init_scene.shape[1], init_scene.shape[2]))

                                current_obj_channel = np.expand_dims(current_obj_channel, axis=0)
                                dynamic_obj_channel = np.expand_dims(dynamic_obj_channel, axis=0)
                                static_obj_channel = np.expand_dims(static_obj_channel, axis=0)
                                current_obj_path_channel = np.expand_dims(current_obj_path_channel, axis=0)

                                # Processed data
                                input_initial_scene = np.concatenate([current_obj_channel, dynamic_obj_channel,
                                                                      static_obj_channel], axis=0)
                                processed_input = np.concatenate([input_initial_scene,
                                                                  current_obj_path_channel], axis=0)

                                procesed_data.append(processed_input.astype(np.uint8))

                            data[i] = None

                    file = gzip.GzipFile(save_path + '/processed_data_sfm1.pickle', 'wb')
                    pickle.dump(procesed_data, file)
                    file.close()

                    procesed_data = T.tensor(procesed_data).float()

                    with open(load_path + '/index.pickle', 'rb') as fp:
                        index = pickle.load(fp)

                    self.train_dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(procesed_data),
                                                                    batchsize, shuffle=False)
                    self.train_index = index

        else:
            # To-do
            pass
        if brute_search:
            if not test:
                # To-do
                pass
            else:
                # To-do
                pass

        with open(f'result/flownet/training/{self.path}/namespace.txt', 'w') as handle:
            handle.write(f"{self.modeltype} {setup} {fold}")

    def load_processed_data_sfm2(self, setup='ball_within_template', fold=0, train_tasks=[], test_tasks=[],
                                 brute_search=False,
                                 n_per_task=1, shuffle=True, test=False, setup_name="all-tasks", proposal_dict=None):
        fold_id = fold
        eval_setup = setup
        width = self.width
        batchsize = 32
        dijkstra_str = "_dijkstra" if self.dijkstra else ""

        if train_tasks and test_tasks:
            train_ids = train_tasks
            test_ids = test_tasks
        else:
            setup_name = "within" if setup == 'ball_within_template' else "cross"
            setup_name = setup_name + dijkstra_str
            if proposal_dict is not None:
                setup_name = setup_name + "_proposals"

        if not test:
            load_path = save_path = f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n"
            if os.path.exists(load_path + "/processed_data_sfm2.pickle"):
                with gzip.open(load_path + '/processed_data_sfm2.pickle', 'rb') as fp:
                    processed_data = pickle.load(fp)
                    X = T.tensor(processed_data).float()

                    with open(load_path + '/index.pickle', 'rb') as fp:
                        index = pickle.load(fp)

                    self.train_dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), batchsize,
                                                                    shuffle=False)
                    self.train_index = index

            else:
                if os.path.exists(load_path + "/data.pickle") and os.path.exists(load_path + "/sfm1_paths.pickle"):
                    with gzip.open(load_path + '/data.pickle', 'rb') as fp:
                        data = pickle.load(fp)

                    with gzip.open(load_path + '/sfm1_paths.pickle', 'rb') as fp:
                        sfm1_paths = pickle.load(fp)

                    processed_data = []

                    obj_idxs = range(6)
                    static_obj_idxs = [3, 5]
                    for i in tqdm(range(len(data[:]))):
                        X = data[i]
                        X = np.true_divide(X, 255.)
                        init_scene = X[:6]
                        paths = sfm1_paths[i]
                        ground_truth_paths = X[6:]

                        for obj_idx in obj_idxs:
                            # skip for static objects
                            if obj_idx in static_obj_idxs:
                                continue

                            # skip if object corresponding to obj_idx is not in scene
                            if not obj_in_scene(init_scene[obj_idx]):
                                continue

                            # Current object
                            current_obj_channel = init_scene[obj_idx]
                            current_obj_path_channel = paths[obj_idx]

                            # Dynamic objects
                            dynamic_obj_idxs = [idx for idx in obj_idxs if idx not in static_obj_idxs
                                                and obj_in_scene(init_scene[idx]) and idx != obj_idx]
                            dynamic_obj_channel = np.max(init_scene[dynamic_obj_idxs], axis=0)
                            dynamic_obj_path_channel = np.max(paths[dynamic_obj_idxs], axis=0)

                            # Static objects
                            static_obj_idxs_in_scene = [idx for idx in static_obj_idxs
                                                        if obj_in_scene(init_scene[idx])]
                            if len(static_obj_idxs_in_scene) > 0:
                                static_obj_channel = np.max(init_scene[static_obj_idxs_in_scene], axis=0)
                            else:
                                static_obj_channel = np.zeros((init_scene.shape[1], init_scene.shape[2]))

                            current_obj_channel = np.expand_dims(current_obj_channel, axis=0)
                            dynamic_obj_channel = np.expand_dims(dynamic_obj_channel, axis=0)
                            static_obj_channel = np.expand_dims(static_obj_channel, axis=0)
                            current_obj_path_channel = np.expand_dims(current_obj_path_channel, axis=0)
                            dynamic_obj_path_channel = np.expand_dims(dynamic_obj_path_channel, axis=0)

                            # Processed data
                            input_channels = np.concatenate([current_obj_channel, dynamic_obj_channel,
                                                             static_obj_channel,
                                                             current_obj_path_channel, dynamic_obj_path_channel],
                                                            axis=0)
                            processed_input = np.concatenate([input_channels,
                                                              ground_truth_paths[obj_idx][None]], axis=0)

                            processed_data.append(processed_input.astype(np.float32))

                        data[i] = None

                    file = gzip.GzipFile(save_path + '/processed_data_sfm2.pickle', 'wb')
                    pickle.dump(processed_data, file)
                    file.close()

                    procesed_data = T.tensor(processed_data).float()

                    with open(load_path + '/index.pickle', 'rb') as fp:
                        index = pickle.load(fp)

                    self.train_dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(procesed_data),
                                                                    batchsize, shuffle=False)
                    self.train_index = index

        else:
            # To-do
            pass
        if brute_search:
            if not test:
                # To-do
                pass
            else:
                # To-do
                pass

        with open(f'result/flownet/training/{self.path}/namespace.txt', 'w') as handle:
            handle.write(f"{self.modeltype} {setup} {fold}")

    def save_sfm1_paths(self, dir, setup, fold, n_per_task):
        MODEL_PATH = f'./saves/flownet/{setup}/SfM1.pt'
        SfMNet1 = self.models["SfM1"]
        SfMNet1.load_state_dict(T.load(MODEL_PATH))
        SfMNet1.to(self.device).eval()

        fold_id = fold
        eval_setup = setup
        width = self.width

        setup_name = "within" if setup == 'ball_within_template' else "cross"

        save_path = f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n"

        # with open("./phyre_get_fold.pickle", "rb") as f:
        #     x = pickle.load(f)
        #
        # train_ids, dev_ids, test_ids = x
        # train_ids += dev_ids + test_ids
        # train_ids = sorted(train_ids)

        train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
        train_ids += dev_ids + test_ids
        train_ids = sorted(train_ids)

        data = make_mono_dataset_2(
            f"data/{setup_name}_fold_{fold_id}_train_{width}xy_{n_per_task}n",
            size=(width, width), tasks=train_ids, n_per_task=n_per_task,
            proposal_dict=None, dijkstra=self.dijkstra, save=False)

        procesed_data = []

        obj_idxs = range(6)
        static_obj_idxs = [3, 5]

        init_scenes = []
        sfm1_paths = []

        for i in tqdm(range(len(data))):
            X = data[i]
            X = np.true_divide(X, 255.)
            init_scene = X[:6]
            sfm1_path = []

            init_scenes.append(init_scene)

            empty_path = np.zeros((1, init_scene.shape[1], init_scene.shape[2]))

            for obj_idx in obj_idxs:
                # skip for static objects
                if obj_idx in static_obj_idxs:
                    sfm1_path.append(empty_path)
                    continue

                # skip if object corresponding to obj_idx is not in scene
                if not obj_in_scene(init_scene[obj_idx]):
                    sfm1_path.append(empty_path)
                    continue

                # Current object
                current_obj_channel = init_scene[obj_idx]

                # Dynamic objects
                dynamic_obj_idxs = [idx for idx in obj_idxs if idx not in static_obj_idxs
                                    and obj_in_scene(init_scene[idx]) and idx != obj_idx]
                dynamic_obj_channel = np.max(init_scene[dynamic_obj_idxs], axis=0)

                # Static objects
                static_obj_idxs_in_scene = [idx for idx in static_obj_idxs
                                            if obj_in_scene(init_scene[idx])]
                if len(static_obj_idxs_in_scene) > 0:
                    static_obj_channel = np.max(init_scene[static_obj_idxs_in_scene], axis=0)
                else:
                    static_obj_channel = np.zeros((init_scene.shape[1], init_scene.shape[2]))

                current_obj_channel = np.expand_dims(current_obj_channel, axis=0)
                dynamic_obj_channel = np.expand_dims(dynamic_obj_channel, axis=0)
                static_obj_channel = np.expand_dims(static_obj_channel, axis=0)

                # Processed data
                input_initial_scene = np.concatenate([current_obj_channel, dynamic_obj_channel,
                                                      static_obj_channel], axis=0)

                predicted_path = SfMNet1(T.tensor(input_initial_scene).float().to(self.device)[None])[0]
                predicted_path = predicted_path.cpu().detach().numpy()
                sfm1_path.append(predicted_path)

            sfm1_paths.append(np.concatenate(sfm1_path, axis=0).astype(np.float32))

        file = gzip.GzipFile(save_path + '/sfm1_paths.pickle', 'wb')
        pickle.dump(sfm1_paths, file)
        file.close()
        print("SfM1 paths saved")

    def train_sfm_sequential(self, setup, epochs=10, width=128, n_per_task=10, fold=0, batchsize=32):
        SfMNet = self.models["SfM_sequential"]
        self.to_train()
        loss_log = []

        opti = T.optim.Adam(SfMNet.parameters(recurse=True), lr=3e-4)

        load_path = f"data/{setup.split('_')[1]}_fold_{fold}_train_{width}xy_{n_per_task}n/data_sequential"
        sequential_data = np.empty((0, 23, width, width), int)

        for task_file in os.listdir(load_path):
            task_file_path = os.path.join(load_path, task_file)
            with gzip.open(task_file_path, 'rb') as fp:
                row = pickle.load(fp)
                row = row.reshape(-1, row.shape[-3],
                                  row.shape[-2],
                                  row.shape[-1])

                sequential_data = np.append(sequential_data, row, axis=0)

        X = T.tensor(sequential_data).float()
        data_loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X),
                                              batchsize, shuffle=True)

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)

                current_obj_frames = X[:, :10]
                next_current_obj_frame = X[:, 10]
                dynamic_obj_frames = X[:, 11: 21]
                static_obj_frame = X[:, 22][:, None]

                sfm_input = T.cat([current_obj_frames, dynamic_obj_frames, static_obj_frame], axis=1)

                # for 4-headed loss
                next_current_obj_frame_128 = next_current_obj_frame
                next_current_obj_frame_64 = self.resize_tensor(next_current_obj_frame, 64).squeeze(1)
                next_current_obj_frame_32 = self.resize_tensor(next_current_obj_frame, 32).squeeze(1)
                next_current_obj_frame_16 = self.resize_tensor(next_current_obj_frame, 16).squeeze(1)
                next_current_obj_frame_all_scales = [next_current_obj_frame_128,
                                                     next_current_obj_frame_64,
                                                     next_current_obj_frame_32,
                                                     next_current_obj_frame_16]

                predicted_path_all_scales = SfMNet(sfm_input)

                loss_all_scales = []
                for predicted_path, gt_path in zip(predicted_path_all_scales,
                                                   next_current_obj_frame_all_scales):
                    loss_one_scale = F.binary_cross_entropy(predicted_path, gt_path.unsqueeze(1))
                    loss_all_scales.append(loss_one_scale)

                loss = sum(loss_all_scales) / len(loss_all_scales)
                print(epoch, i, loss_all_scales[0].item())

                if i % 500 == 0:
                    empty_channel = T.zeros((32, 1, width, width)).to(self.device)

                    current_obj_next_state = T.cat([predicted_path_all_scales[0], empty_channel,
                                                    next_current_obj_frame[:, None]], axis=1).permute(0, 2, 3, 1)

                    current_obj_path = np.max(current_obj_frames.cpu().numpy(), axis=1)[:, None]
                    dynamic_obj_path = np.max(dynamic_obj_frames.cpu().numpy(), axis=1)[:, None]
                    static_obj_path = static_obj_frame.cpu().numpy()

                    init_path = np.concatenate([current_obj_path, dynamic_obj_path, static_obj_path],
                                               axis=1)

                    batch_images = [np.moveaxis(init_path, 1, -1), current_obj_next_state.cpu().detach().numpy()]

                    vis_pred_path(batch_images, f'./proposalNet_result/{setup}/SfM_sequential', str(epoch) + "_" + str(i))

                    loss_log.append([str(epoch) + "_" + str(i), loss_all_scales[0].item()])

                opti.zero_grad()
                loss.backward()
                opti.step()

        log_loss_df = pd.DataFrame(loss_log, columns=['Epoch and iteration', 'loss'])
        log_loss_df.to_csv(f'./result/flownet/inspect/standard/default/{setup}/loss.csv', index=False)

    def train_sfm1(self, setup, epochs=10):
        SfMNet1 = self.models["SfM1"]
        self.to_train()
        loss_log = []

        data_loader = self.train_dataloader

        opti = T.optim.Adam(SfMNet1.parameters(recurse=True), lr=3e-3)

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)

                init_scene = X[:, :3]
                current_obj_path_channel = X[:, 3]

                # for 4-headed loss
                current_obj_path_channel_128 = current_obj_path_channel
                current_obj_path_channel_64 = self.resize_tensor(current_obj_path_channel, 64).squeeze(1)
                current_obj_path_channel_32 = self.resize_tensor(current_obj_path_channel, 32).squeeze(1)
                current_obj_path_channel_16 = self.resize_tensor(current_obj_path_channel, 16).squeeze(1)
                current_obj_path_channel_all_scales = [current_obj_path_channel_128,
                                                       current_obj_path_channel_64,
                                                       current_obj_path_channel_32,
                                                       current_obj_path_channel_16]

                predicted_path_all_scales = SfMNet1(init_scene)

                loss_all_scales = []
                for predicted_path, gt_path in zip(predicted_path_all_scales,
                                                   current_obj_path_channel_all_scales):
                    loss_one_scale = F.binary_cross_entropy(predicted_path, gt_path.unsqueeze(1))
                    loss_all_scales.append(loss_one_scale)

                loss = sum(loss_all_scales) / len(loss_all_scales)
                print(epoch, i, loss_all_scales[0].item())

                # visualisation
                if i % 100 == 0:
                    empty_channel = T.zeros((32, 1, init_scene.size(2), init_scene.size(3))).to(self.device)

                    current_obj_path = T.cat([predicted_path_all_scales[0], empty_channel,
                                              empty_channel], axis=1).permute(0, 2, 3, 1)

                    batch_images = [init_scene.permute(0, 2, 3, 1).cpu().detach().numpy(),
                                    current_obj_path.cpu().detach().numpy()]

                    vis_pred_path(batch_images, f'./proposalNet_result/{setup}/SfM1', str(epoch) + "_" + str(i))
                    loss_log.append([str(epoch) + "_" + str(i), loss_all_scales[0].item()])

                opti.zero_grad()
                loss.backward()
                opti.step()

        log_loss_df = pd.DataFrame(loss_log, columns=['Epoch and iteration', 'loss'])
        log_loss_df.to_csv(f'./result/flownet/inspect/standard/default/{setup}/loss.csv', index=False)

    def train_sfm2(self, setup, epochs=10):
        SfMNet2 = self.models["SfM2"]
        self.to_train()
        loss_log = []

        data_loader = self.train_dataloader

        opti = T.optim.Adam(SfMNet2.parameters(recurse=True), lr=3e-3)

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)

                sfm2_input = X[:, :5]
                current_obj_path_channel = X[:, 5]

                # for 4-headed loss
                current_obj_path_channel_128 = current_obj_path_channel
                current_obj_path_channel_64 = self.resize_tensor(current_obj_path_channel, 64).squeeze(1)
                current_obj_path_channel_32 = self.resize_tensor(current_obj_path_channel, 32).squeeze(1)
                current_obj_path_channel_16 = self.resize_tensor(current_obj_path_channel, 16).squeeze(1)
                current_obj_path_channel_all_scales = [current_obj_path_channel_128,
                                                       current_obj_path_channel_64,
                                                       current_obj_path_channel_32,
                                                       current_obj_path_channel_16]

                predicted_path_all_scales = SfMNet2(sfm2_input)

                loss_all_scales = []
                for predicted_path, gt_path in zip(predicted_path_all_scales,
                                                   current_obj_path_channel_all_scales):
                    loss_one_scale = F.binary_cross_entropy(predicted_path, gt_path.unsqueeze(1))
                    loss_all_scales.append(loss_one_scale)

                loss = sum(loss_all_scales) / len(loss_all_scales)
                print(epoch, i, loss_all_scales[0].item())

                # visualisation
                if i % 100 == 0:
                    empty_channel = T.zeros((32, 1, sfm2_input.size(2), sfm2_input.size(3))).to(self.device)

                    current_obj_path = T.cat([predicted_path_all_scales[0], empty_channel,
                                              empty_channel], axis=1).permute(0, 2, 3, 1)

                    batch_images = [sfm2_input[:, :3].permute(0, 2, 3, 1).cpu().detach().numpy(),
                                    current_obj_path.cpu().detach().numpy()]

                    vis_pred_path(batch_images, f'./proposalNet_result/{setup}/SfM2/', str(epoch) + "_" + str(i))
                    loss_log.append([str(epoch) + "_" + str(i), loss_all_scales[0].item()])

                opti.zero_grad()
                loss.backward()
                opti.step()

        log_loss_df = pd.DataFrame(loss_log, columns=['Epoch and iteration', 'loss'])
        log_loss_df.to_csv(f'./result/flownet/inspect/standard/default/{setup}/loss_sfm2.csv', index=False)

    def train_supervised(self, train_mode='CONS', epochs=10):
        self.to_train()
        # self.load_models()
        # self.models["tar_net1"].load_state_dict(T.load('./saves/flownet/standard_within_0/tar_net_py.pt', map_location=T.device("cuda")))
        data_loader = self.train_dataloader
        tar_net = self.models["tar_net"]
        base_net = self.models["base_net"]
        act_net = self.models["act_net"]
        ext_net = self.models["ext_net"]
        # tar_net2 = self.models["tar_net2"]
        log = []
        base_loss_log = []
        self.logger["gen-train-loss"] = log

        opti = T.optim.Adam(chain(act_net.parameters(recurse=True),
                                  ext_net.parameters(recurse=True),
                                  base_net.parameters(recurse=True),
                                  tar_net.parameters(recurse=True)),
                            lr=3e-3)

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X.to(self.device)
                # Prepare Data
                action_balls = X[:, 0]
                init_scenes = X[:, 1:6]
                base_paths = X[:, 6]

                if self.modeltype == "SfMLearner":
                    base_paths_128 = base_paths
                    base_paths_64 = resize_tensor(base_paths, 64)
                    base_paths_32 = resize_tensor(base_paths, 32)
                    base_paths_16 = resize_tensor(base_paths, 16)
                    base_paths_all_sizes = base_paths_128, base_paths_64, base_paths_32, base_paths_16
                    base_paths = base_paths_128

                target_paths = X[:, 7]
                goal_paths = X[:, 8]
                action_paths = X[:, 9]
                if self.dijkstra:
                    dist_map = X[:, None, 10]
                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                if train_mode == 'MIX':
                    modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                else:
                    modus = train_mode

                # Forward Pass
                if self.dijkstra:
                    target_pred = tar_net(T.cat((init_scenes, dist_map), dim=1))
                else:
                    target_pred = tar_net(init_scenes)

                # tar_net2_input = target_pred1.clone().detach()
                # target_pred2 = tar_net2(T.cat((init_scenes, tar_net2_input), dim=1))

                base_pred_all_sizes = base_net(init_scenes)
                if self.modeltype == "SfMLearner":
                    base_pred = base_pred_all_sizes[0]
                else:
                    base_pred = base_pred_all_sizes

                if modus == 'GT':
                    action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
                elif modus == 'CONS':
                    action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
                elif modus == 'COMB':
                    action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                elif modus == 'END':
                    action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                    ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

                if not i % 100:
                    os.makedirs(f'result/flownet/training/{self.path}/{self.run}', exist_ok=True)
                    # print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                    # text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action', 'base', 'target', 'action', 'red ball']
                    # vis_batch(print_batch.cpu(), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}', text=text)

                    # sum_batch = T.cat((base_paths[:,None]+base_pred, target_paths[:,None]+target_pred, action_paths[:,None]+action_pred, action_balls[:,None]+ball_pred), dim=1).detach().abs()/2
                    # text = ['base\npaths', 'target\npaths', 'action\npaths', 'action\nballs']
                    # vis_batch(sum_batch.cpu(), f'result/flownet/training/{self.path}', f'poch_{epoch}_{i}_sum', text=text)

                    background = init_scenes[:, 3:].sum(dim=1)[:, None]
                    background = background / max(background.max(), 1)
                    diff_batch = T.cat((
                        T.stack((background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                        T.stack((base_pred, base_paths[:, None], T.zeros_like(base_pred)), dim=-1),
                        T.stack((target_pred, target_paths[:, None], T.zeros_like(target_pred)), dim=-1),
                        T.stack((action_pred, action_paths[:, None], T.zeros_like(action_pred)), dim=-1),
                        T.stack((ball_pred, action_balls[:, None], T.zeros_like(ball_pred)), dim=-1),
                        T.stack((ball_pred + background, init_scenes[:, None, 1] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                        T.stack((action_balls[:, None] + background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)),
                        dim=1).detach()
                    diff_batch = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch = white
                    text = ['initial\nscene', 'base\npaths', 'target\npaths', 'action\npaths', 'action\nballs',
                            'injected\nscene', 'GT\nscene']
                    if self.dijkstra:
                        print_dms = T.stack((dist_map, dist_map, dist_map), dim=-1).cpu()
                        diff_batch = T.cat((diff_batch, print_dms), dim=1)
                        text.append("dijkstra")
                    vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/training/{self.path}/{self.run}',
                              f'poch_{epoch}_{i}_generator', text=text)
                # plt.show()

                # Loss
                tar_loss = F.binary_cross_entropy(target_pred, target_paths[:, None])
                # tar_loss2 = F.binary_cross_entropy(target_pred2, target_paths[:, None])
                act_loss = F.binary_cross_entropy(action_pred, action_paths[:, None])
                ball_loss = F.binary_cross_entropy(ball_pred, action_balls[:, None])

                if self.modeltype == "SfMLearner":
                    base_losses = []
                    for pred, gt in zip(base_pred_all_sizes, base_paths_all_sizes):
                        base_losses.append(F.binary_cross_entropy(pred, gt[:, None]))
                    base_loss_log.append(base_losses[0].item())
                    base_loss = sum(base_losses) / len(base_losses)
                else:
                    base_loss = F.binary_cross_entropy(base_pred, base_paths[:, None])

                if modus == 'END':
                    loss = ball_loss
                else:
                    loss = ball_loss + tar_loss + act_loss + base_loss
                print(epoch, i, loss.item(), end='\r')

                # Backward Pass
                opti.zero_grad()
                loss.backward()
                opti.step()
                log.append(loss.item())
                # base_loss_log.append(base_loss.item())

        with open(f'result/flownet/inspect/{self.path}/{self.run}/loss.txt', 'a+') as fp:
            fp.write(f"avg-loss {sum(log) / len(log)} ")
            fp.write(f"base loss {sum(base_loss_log) / len(base_loss_log)} ")
            fp.write(f"avg-ten-smallest-losses {sum(sorted(log)[:10]) / 10}\n")

    def train_brute_search(self, train_mode='CONS', epochs=10):
        self.to_train()
        data_loader = self.train_dataloader
        fail_loader = self.failed_dataloader
        base_net = self.models["base_net"]
        act_net = self.models["act_net"]
        tar_net = self.models["tar_net"]
        ext_net = self.models["ext_net"]
        complete_percentage = []

        opti = T.optim.Adam(chain(tar_net.parameters(recurse=True),
                                  act_net.parameters(recurse=True),
                                  ext_net.parameters(recurse=True),
                                  base_net.parameters(recurse=True)),
                            lr=3e-3)

        for epoch in range(epochs):
            percentage = []
            for i, ((X,), (Z,)) in enumerate(zip(data_loader, fail_loader)):
                last_index = min(X.shape[0], Z.shape[0])
                X, Z = X[:last_index].to(self.device), Z[:last_index].to(self.device)
                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)

                # Prepare Data
                solve_scenes = X[:, :6]
                solve_base_paths = X[:, 6]
                solve_target_paths = X[:, 7]
                solve_action_paths = X[:, 9]
                solve_goal_paths = X[:, 8]
                fail_scenes = Z[:, [0, 1, 2, 3, 4, 5]]
                fail_base_paths = Z[:, 6]
                fail_target_paths = Z[:, 7]
                fail_action_paths = Z[:, 9]
                fail_goal_paths = Z[:, 8]

                init_scenes = T.cat((solve_scenes, fail_scenes), dim=0)
                target_paths = T.cat((solve_target_paths, fail_target_paths), dim=0)
                base_paths = T.cat((solve_base_paths, fail_base_paths), dim=0)
                action_paths = T.cat((solve_action_paths, fail_action_paths), dim=0)
                goal_paths = T.cat((solve_goal_paths, fail_goal_paths), dim=0)

                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                if train_mode == 'MIX':
                    modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                else:
                    modus = train_mode

                # Forward Pass
                base_pred = base_net(init_scenes)
                if modus == 'GT':
                    action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                    confidence = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
                elif modus == 'CONS':
                    action_pred = act_net(T.cat((init_scenes, base_pred.detach()), dim=1))
                    target_pred = tar_net(T.cat((init_scenes, base_pred.detach(), action_pred.detach()), dim=1))
                    confidence = ext_net(
                        T.cat((init_scenes, base_pred.detach(), target_pred.detach(), action_pred.detach()), dim=1))
                elif modus == 'COMB':
                    action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                    target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                    confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))
                elif modus == 'END':
                    action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                    target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                    confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))

                if not i % 100:
                    os.makedirs(f'result/flownet/training/{self.path}', exist_ok=True)
                    print_batch = T.cat((init_scenes, goal_paths[:, None], base_paths[:, None], base_pred,
                                         target_paths[:, None], target_pred,
                                         action_paths[:, None], action_pred,
                                         T.ones_like(base_pred) * confidence[:, :, None, None],
                                         T.ones_like(base_pred) * conf_target[:, None, None, None]), dim=1).detach()
                    text = [
                        'GT\n red ball',
                        'GT\ngreen ball',
                        'GT\nblue dynamic\nobject',
                        'GT\nblue static\nobject',
                        'GT\ngrey dynamic\nobjects',
                        'GT\nblack static\nobjects',
                        'GT\nblue dynamic\npath',
                        'GT\ngreen ball\nbase path',
                        'predicted\ngreen ball\nbase path',
                        'GT\ngreen ball\ntarget path',
                        'predicted\ngreen ball\ntarget path',
                        'GT\nred ball\naction path',
                        'predicted\nred ball\naction path',
                        'predicted\nconfidence',
                        'GT\nconfidence']
                    # print_batch = T.cat((T.cat((X,Z), dim=0), base_pred, target_pred, action_pred, T.ones_like(base_pred)*confidence[:,:,None,None]), dim=1).detach()
                    # text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\nGT path', 'target\nGT path', 'goal GT\nnot used', 'action\nGT path', 'base\npath pred', 'target\npath pred', 'action\npath pred', 'conf']
                    vis_batch(print_batch.cpu(), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}',
                              text=text)

                    background = init_scenes[:, 4:].sum(dim=1)[:, None]
                    background = background / max(background.max(), 1)
                    tmp_conf = T.ones_like(base_pred) * confidence[:, :, None, None]
                    diff_batch = T.cat((
                        T.stack((init_scenes[:, None, 0] + background, init_scenes[:, None, 1] + background,
                                 init_scenes[:, None, 2] + init_scenes[:, None, 3] + background), dim=-1),
                        T.stack((base_pred, base_paths[:, None], T.zeros_like(base_pred)), dim=-1),
                        T.stack((target_pred, target_paths[:, None], T.zeros_like(target_pred)), dim=-1),
                        T.stack((action_pred, action_paths[:, None], T.zeros_like(action_pred)), dim=-1),
                        T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1)),
                        dim=1).detach()
                    diff_batch = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch = white
                    text = ['scene', 'base\npaths', 'target\npaths', 'action\npaths', 'confidence']
                    vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/training/{self.path}/{self.run}',
                              f'poch_{epoch}_{i}_diff', text=text)
                # plt.show()

                # Loss
                tar_loss = F.binary_cross_entropy(target_pred, target_paths[:, None])
                act_loss = F.binary_cross_entropy(action_pred, action_paths[:, None])
                conf_loss = F.binary_cross_entropy(confidence[:, 0], conf_target)
                base_loss = F.binary_cross_entropy(base_pred, base_paths[:, None])

                print(epoch, i, (confidence[:, 0].round() == conf_target).float().sum().item() / confidence.shape[0],
                      "classified correctly", end='\r')
                percentage.append((confidence[:, 0].round() == conf_target).float().sum().item() / confidence.shape[0])

                if modus == 'END':
                    loss = conf_loss
                else:
                    loss = conf_loss + tar_loss + act_loss + base_loss
                # print(epoch, i, loss.item(), end='\r')

                # Backward Pass
                opti.zero_grad()
                loss.backward()
                opti.step()

            print("epoch {epoch}: avg correct classified percentage:", sum(percentage) / len(percentage))
            complete_percentage.append(sum(percentage) / len(percentage))
        print(complete_percentage)

    def train_combi(self, train_mode='CONS', epochs=10):
        self.to_train()
        data_loader = self.train_dataloader
        fail_loader = self.failed_dataloader
        sim_net = self.models["sim_net"]
        comb_net = self.models["comb_net"]
        success_net = self.models["success_net"]
        accuracy = 0

        os.makedirs(f'result/flownet/training/{self.path}/{self.run}', exist_ok=True)
        fp = open(f'result/flownet/training/{self.path}/{self.run}/accuracy.txt', 'w')

        opti = T.optim.Adam(chain(sim_net.parameters(recurse=True),
                                  comb_net.parameters(recurse=True),
                                  success_net.parameters(recurse=True)),
                            lr=3e-3)

        for epoch in range(epochs):
            for i, ((X,), (Z,)) in enumerate(zip(data_loader, fail_loader)):
                last_index = min(X.shape[0], Z.shape[0])
                X, Z = X[:last_index].to(self.device), Z[:last_index].to(self.device)

                # Prepare Data
                solve_scenes = X[:, :6]
                solve_base_paths = X[:, 6]
                solve_target_paths = X[:, 7]
                solve_action_paths = X[:, 9]
                solve_goal_paths = X[:, 8]
                fail_scenes = Z[:, [0, 1, 2, 3, 4, 5]]
                fail_base_paths = Z[:, 6]
                fail_target_paths = Z[:, 7]
                fail_action_paths = Z[:, 9]
                fail_goal_paths = Z[:, 8]

                init_scenes = T.cat((solve_scenes, fail_scenes), dim=0)
                target_paths = T.cat((solve_target_paths, fail_target_paths), dim=0)
                base_paths = T.cat((solve_base_paths, fail_base_paths), dim=0)
                action_paths = T.cat((solve_action_paths, fail_action_paths), dim=0)
                goal_paths = T.cat((solve_goal_paths, fail_goal_paths), dim=0)

                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)

                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                if train_mode == 'MIX':
                    modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                else:
                    modus = train_mode

                # Forward Pass
                red = sim_net(init_scenes)  # red path
                green = sim_net(init_scenes[:, [1, 0, 2, 3, 4, 5]])  # green path
                blue = sim_net(init_scenes[:, [2, 0, 1, 3, 4, 5]])  # blue path
                comb = comb_net(init_scenes)
                if modus == 'GT':
                    confidence = success_net(
                        T.cat((init_scenes, action_paths[:, None], target_paths[:, None], goal_paths[:, None]), dim=1))
                elif modus == 'CONS':
                    confidence = success_net(T.cat((init_scenes, red.detach(), green.detach(), blue.detach()), dim=1))
                elif modus == 'COMB':
                    confidence = success_net(T.cat((init_scenes, red, green, blue), dim=1))
                elif modus == 'END':
                    confidence = success_net(T.cat((init_scenes, red, green, blue), dim=1))

                if not i % 100:
                    # os.makedirs(f'result/flownet/training/{self.path}', exist_ok=True)
                    # print_batch = T.cat((T.cat((X,Z), dim=0), prediction, T.ones_like(base_paths[:,None])*confidence[:,:,None,None]), dim=1).detach()
                    # text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\nGT path', 'target\nGT path', 'action\nGT path', 'goal\nGT path', 'target\npath pred', 'action\npath pred', 'goal\npath pred', 'conf']
                    # vis_batch(print_batch.cpu(), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}_pipe', text=text)

                    background = init_scenes[:, 4:].sum(dim=1)[:, None]
                    background = background / max(background.max(), 1)
                    tmp_conf = T.ones_like(base_paths[:, None]) * confidence[:, :, None, None]
                    gt_conf = T.ones_like(base_paths[:, None]) * conf_target[:, None, None, None]
                    scene = T.stack((init_scenes[:, None, 0] + background, init_scenes[:, None, 1] + background,
                                     init_scenes[:, None, 2] + init_scenes[:, None, 3] + background), dim=-1)
                    diff_batch = T.cat((
                        scene,
                        scene + T.stack((red, green, blue), dim=-1),
                        scene + T.stack((action_paths[:, None], target_paths[:, None], goal_paths[:, None]), dim=-1),
                        scene + T.stack((comb[:, None, 0], comb[:, None, 1], comb[:, None, 2]), dim=-1),
                        T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1),
                        T.stack((1 - gt_conf, 1 - gt_conf, 1 - gt_conf), dim=-1)),
                        dim=1).detach()
                    diff_batch = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch = white
                    text = ['scene', 'seperately\npredicted\npaths', 'GT paths', 'combined\npredicted\npaths',
                            'predicted\nconfidence', 'GT\nconfidence']
                    vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/training/{self.path}/{self.run}',
                              f'poch_{epoch}_{i}_predictor', text=text)
                # plt.show()

                # Loss
                conf_loss = F.binary_cross_entropy(confidence[:, 0], conf_target)

                red_loss = F.binary_cross_entropy(red, action_paths[:, None])
                green_loss = F.binary_cross_entropy(green, target_paths[:, None])
                blue_loss = F.binary_cross_entropy(blue, goal_paths[:, None])
                comb_loss = F.binary_cross_entropy(comb, T.cat(
                    (action_paths[:, None], target_paths[:, None], goal_paths[:, None]), dim=1))
                # print((red<0).any().item(), (red>1).any().item(), (green<0).any().item(), (green>1).any().item(), (blue<0).any().item(), (blue>1).any().item())

                acc = (confidence[:, 0].round() == conf_target).float().sum().item() / confidence.shape[0]
                accuracy = 0.9 * accuracy + 0.1 * acc
                if not i % 10:
                    print(epoch, i, accuracy, "classified correctly", end='\r')
                    fp.write(f"{accuracy}\n")

                if modus == 'END':
                    loss = conf_loss
                else:
                    loss = conf_loss + red_loss + green_loss + blue_loss + comb_loss
                # print(epoch, i, loss.item(), end='\r')

                # Backward Pass
                opti.zero_grad()
                loss.backward()
                opti.step()

        fp.close()

    def inspect_combi(self, train_mode='CONS', epochs=10):
        self.to_train()
        data_loader = self.test_dataloader
        fail_loader = self.failed_test_dataloader
        sim_net = self.models["sim_net"]
        comb_net = self.models["comb_net"]
        success_net = self.models["success_net"]
        accuracy = 0

        percentage = []
        precision = []
        recall = []

        opti = T.optim.Adam(chain(sim_net.parameters(recurse=True),
                                  comb_net.parameters(recurse=True),
                                  success_net.parameters(recurse=True)),
                            lr=3e-3)

        for epoch in range(1):
            for i, ((X,), (Z,)) in enumerate(zip(data_loader, fail_loader)):
                last_index = min(X.shape[0], Z.shape[0])
                X, Z = X[:last_index].to(self.device), Z[:last_index].to(self.device)

                # Prepare Data
                solve_scenes = X[:, :6]
                solve_base_paths = X[:, 6]
                solve_target_paths = X[:, 7]
                solve_action_paths = X[:, 9]
                solve_goal_paths = X[:, 8]
                fail_scenes = Z[:, [0, 1, 2, 3, 4, 5]]
                fail_base_paths = Z[:, 6]
                fail_target_paths = Z[:, 7]
                fail_action_paths = Z[:, 9]
                fail_goal_paths = Z[:, 8]

                init_scenes = T.cat((solve_scenes, fail_scenes), dim=0)
                target_paths = T.cat((solve_target_paths, fail_target_paths), dim=0)
                base_paths = T.cat((solve_base_paths, fail_base_paths), dim=0)
                action_paths = T.cat((solve_action_paths, fail_action_paths), dim=0)
                goal_paths = T.cat((solve_goal_paths, fail_goal_paths), dim=0)

                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)

                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                if train_mode == 'MIX':
                    modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                else:
                    modus = train_mode

                # Forward Pass
                red = sim_net(init_scenes)  # red path
                green = sim_net(init_scenes[:, [1, 0, 2, 3, 4, 5]])  # green path
                blue = sim_net(init_scenes[:, [2, 0, 1, 3, 4, 5]])  # blue path
                comb = comb_net(init_scenes)
                if modus == 'GT':
                    confidence = success_net(
                        T.cat((init_scenes, action_paths[:, None], target_paths[:, None], goal_paths[:, None]), dim=1))
                elif modus == 'CONS':
                    confidence = success_net(T.cat((init_scenes, red.detach(), green.detach(), blue.detach()), dim=1))
                elif modus == 'COMB':
                    confidence = success_net(T.cat((init_scenes, red, green, blue), dim=1))
                elif modus == 'END':
                    confidence = success_net(T.cat((init_scenes, red, green, blue), dim=1))

                if not i % 100:
                    # os.makedirs(f'result/flownet/training/{self.path}', exist_ok=True)
                    # print_batch = T.cat((T.cat((X,Z), dim=0), prediction, T.ones_like(base_paths[:,None])*confidence[:,:,None,None]), dim=1).detach()
                    # text = ['red GT', 'green GT', 'blue GT', 'blue GT', 'grey GT', 'black GT', 'base\nGT path', 'target\nGT path', 'action\nGT path', 'goal\nGT path', 'target\npath pred', 'action\npath pred', 'goal\npath pred', 'conf']
                    # vis_batch(print_batch.cpu(), f'result/flownet/training/{self.path}/{self.run}', f'poch_{epoch}_{i}_pipe', text=text)

                    background = init_scenes[:, 4:].sum(dim=1)[:, None]
                    background = background / max(background.max(), 1)
                    tmp_conf = T.ones_like(base_paths[:, None]) * confidence[:, :, None, None]
                    gt_conf = T.ones_like(base_paths[:, None]) * conf_target[:, None, None, None]
                    scene = T.stack((init_scenes[:, None, 0] + background, init_scenes[:, None, 1] + background,
                                     init_scenes[:, None, 2] + init_scenes[:, None, 3] + background), dim=-1)
                    diff_batch = T.cat((
                        scene,
                        scene + T.stack((red, green, blue), dim=-1),
                        scene + T.stack((action_paths[:, None], target_paths[:, None], goal_paths[:, None]), dim=-1),
                        scene + T.stack((comb[:, None, 0], comb[:, None, 1], comb[:, None, 2]), dim=-1),
                        T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1),
                        T.stack((1 - gt_conf, 1 - gt_conf, 1 - gt_conf), dim=-1)),
                        dim=1).detach()
                    diff_batch = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch = white
                    text = ['scene', 'seperately\npredicted\npaths', 'GT paths', 'combined\npredicted\npaths',
                            'predicted\nconfidence', 'GT\nconfidence']
                    vis_batch(self.cut_off(diff_batch.cpu()), f'result/flownet/training/{self.path}/{self.run}',
                              f'poch_{epoch}_{i}_predictor', text=text)
                # plt.show()

                # Loss)
                conf_loss = F.binary_cross_entropy(confidence[:, 0], conf_target)

                red_loss = F.binary_cross_entropy(red, action_paths[:, None])
                green_loss = F.binary_cross_entropy(green, target_paths[:, None])
                blue_loss = F.binary_cross_entropy(blue, goal_paths[:, None])
                comb_loss = F.binary_cross_entropy(comb, T.cat(
                    (action_paths[:, None], target_paths[:, None], goal_paths[:, None]), dim=1))
                # print((red<0).any().item(), (red>1).any().item(), (green<0).any().item(), (green>1).any().item(), (blue<0).any().item(), (blue>1).any().item())

                # EVAL PERFORMANCE:
                conf_pred = confidence[:, 0].round()
                percentage.append((conf_pred == conf_target).float().sum().item() / confidence.shape[0])
                recall.append(conf_pred[conf_target == 1].sum() / conf_target.sum())
                precision.append(conf_target[conf_pred == 1].sum() / conf_pred.sum())

                acc = (confidence[:, 0].round() == conf_target).float().sum().item() / confidence.shape[0]
                accuracy = 0.9 * accuracy + 0.1 * acc
                if not i % 10:
                    print(epoch, i, accuracy, "classified correctly", end='\r')

                if modus == 'END':
                    loss = conf_loss
                else:
                    loss = conf_loss + red_loss + green_loss + blue_loss + comb_loss
                # print(epoch, i, loss.item(), end='\r')

                # Backward Pass
                opti.zero_grad()
                loss.backward()
                opti.step()

        accuracy = sum(percentage) / len(percentage)
        recall = sum(recall) / len(recall)
        precision = sum(precision) / len(precision)
        print("AVERAGES:  accuracy:", accuracy, "recall", recall, "precision", precision)
        print(percentage)
        os.makedirs(f'result/solver/result/{self.path}/{self.run}', exist_ok=True)
        with open(f'result/solver/result/{self.path}/{self.run}/classification_{eval_setup}_fold_{fold}.txt',
                  'a') as fp:
            fp.write(f"\naccuracy {accuracy}\nrecall {recall}\nprecision {precision}")

    def inspect_supervised(self, eval_setup, fold, train_mode='CONS', epochs=1):
        self.to_train()
        data_loader = self.test_dataloader
        index = self.test_index
        tar_net = self.models["tar_net"]
        base_net = self.models["base_net"]
        act_net = self.models["act_net"]
        ext_net = self.models["ext_net"]
        log = []
        self.logger["gen-inspect-loss"] = log

        """
        task_index = dict()
        for key in index:
            task_index[index[key]] = key
            #print(index[key])
        """

        for epoch in range(epochs):
            for i, (X,) in enumerate(data_loader):
                X = X[:X.shape[0] // 4].to(self.device)
                # Prepare Data
                action_balls = X[:, 0]
                init_scenes = X[:, 1:6]
                base_paths = X[:, 6]
                target_paths = X[:, 7]
                goal_paths = X[:, 8]
                action_paths = X[:, 9]
                if self.dijkstra:
                    dist_map = X[:, None, 10]

                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

                # FORWARD PASS
                with T.no_grad():
                    if train_mode == 'MIX':
                        modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                    else:
                        modus = train_mode

                    # Forward Pass
                    if self.dijkstra:
                        target_pred = tar_net(T.cat((init_scenes, dist_map), dim=1))
                    else:
                        target_pred = tar_net(init_scenes)
                    base_pred = base_net(init_scenes)
                    if modus == 'GT':
                        action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
                    elif modus == 'CONS':
                        action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
                    elif modus == 'COMB':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
                    elif modus == 'END':
                        action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                        ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

                    tar_loss = F.binary_cross_entropy(target_pred, target_paths[:, None])
                    act_loss = F.binary_cross_entropy(action_pred, action_paths[:, None])
                    ball_loss = F.binary_cross_entropy(ball_pred, action_balls[:, None])
                    base_loss = F.binary_cross_entropy(base_pred, base_paths[:, None])
                    if modus == 'END':
                        loss = ball_loss
                    else:
                        loss = ball_loss + tar_loss + act_loss + base_loss
                    print(epoch, i, loss.item(), end='\r')
                    log.append(loss.item())

                # VISUALIZATION
                if self.viz and not i % self.viz:
                    os.makedirs(f'result/flownet/inspect/{self.path}/{self.run}', exist_ok=True)
                    # Z = X.cpu()
                    # vis_batch(T.stack((Z, T.zeros_like(Z), T.zeros_like(Z)), dim=-1), "result/test", "color", text=["hello!"])

                    print_batch = T.cat((init_scenes, goal_paths[:, None], base_paths[:, None], base_pred,
                                         target_paths[:, None], target_pred,
                                         action_paths[:, None], action_pred, ball_pred, action_balls[:, None]),
                                        dim=1).detach()
                    rows = [str(num) for num in range(X.shape[0])]
                    text = [
                        'GT\ngreen ball',
                        'GT\nblue dynamic\nobject',
                        'GT\nblue static\nobject',
                        'GT\ngrey dynamic\nobjects',
                        'GT\nblack static\nobjects',
                        'GT\nblue dynamic\npath',
                        'GT\ngreen ball\nbase path',
                        'predicted\ngreen ball\nbase path',
                        'GT\ngreen ball\ntarget path',
                        'predicted\ngreen ball\ntarget path',
                        'GT\nred ball\naction path',
                        'predicted\nred ball\naction path',
                        'predicted\nred ball',
                        'GT\nred ball']
                    vis_batch(print_batch.cpu(),
                              f'result/flownet/inspect/{self.path}/{self.run}/{eval_setup}_fold_{fold}',
                              f'poch_{epoch}_{i}', text=text, rows=rows)

                    # sum_batch = T.cat((base_paths[:,None]+base_pred, target_paths[:,None]+target_pred, action_paths[:,None]+action_pred, action_balls[:,None]+ball_pred), dim=1).detach().abs()/2
                    # text = ['base\npaths', 'target\npaths', 'action\npaths', 'action\nballs']
                    # vis_batch(sum_batch.cpu(), f'result/flownet/inspect/{self.path}/{eval_setup}_fold_{fold}', f'poch_{epoch}_{i}_sum', text=text)

                    # Extract action and draw:
                    drawings = T.zeros_like(ball_pred)
                    print("drawing balls for batch", i)
                    for b_idx, ball in enumerate(ball_pred[:, 0].cpu()):
                        mask = np.max(init_scenes[b_idx].cpu().numpy(), axis=0)
                        a = grow_action_vector(ball, r_fac=self.r_fac, check_border=True, num_seeds=self.num_seeds,
                                               mask=mask)
                        # print(a)
                        drawn = draw_ball(self.width, a[0], a[1], a[2], invert_y=True)
                        drawings[b_idx, 0] = drawn

                    background = init_scenes[:, 3:].sum(dim=1)[:, None]
                    background = background / max(background.max(), 1)
                    diff_batch = T.cat((
                        T.stack((background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                        0.5 * T.stack((base_pred, base_paths[:, None], base_paths[:, None] + base_pred), dim=-1),
                        0.5 * T.stack((target_pred, target_paths[:, None], target_paths[:, None] + target_pred),
                                      dim=-1),
                        0.5 * T.stack((action_pred, action_paths[:, None], action_paths[:, None] + action_pred),
                                      dim=-1),
                        T.stack((ball_pred, action_balls[:, None], T.zeros_like(ball_pred)), dim=-1),
                        T.stack((ball_pred + background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                        T.stack((drawings + ball_pred + background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                        # T.stack((drawings+background, init_scenes[:,None,0]+background, init_scenes[:,None,1]+init_scenes[:,None,2]+background),dim=-1),
                        T.stack((action_balls[:, None] + background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)),
                        dim=1).detach()
                    diff_batch = self.cut_off(diff_batch.cpu())
                    white = T.ones_like(diff_batch)
                    white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch = white
                    text = ['initial\nscene', 'base paths\nGT: blue-green\npredict: purple',
                            'target paths\nGT: blue-green\npredict: purple',
                            'action paths\nGT: blue-green\npredict: purple',
                            'action balls\nGT: blue-green\npredict: purple', 'action ball\nprediction\nin scene',
                            'prediction\nand resulting\naction in scene', 'GT scene']

                    if self.dijkstra:
                        print_dms = T.stack((dist_map, dist_map, dist_map), dim=-1)
                        diff_batch = T.cat((diff_batch, print_dms), dim=1)
                        text.append("dijkstra")
                    vis_batch(self.cut_off(diff_batch.cpu()),
                              f'result/flownet/inspect/{self.path}/{self.run}/{eval_setup}_fold_{fold}',
                              f'poch_{epoch}_{i}_diff', text=text, rows=rows)
                    scene = T.stack((background, init_scenes[:, None, 0] + background,
                                     init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)
                    diff_batch_left = T.cat((
                        scene,
                        scene + T.stack((base_pred * 0, base_paths[:, None], 0 * base_paths[:, None]), dim=-1),
                        scene + T.stack((0 * target_pred, target_paths[:, None], 0 * target_paths[:, None]), dim=-1),
                        scene + T.stack((action_paths[:, None], 0 * action_pred, 0 * action_paths[:, None]), dim=-1),
                        scene + T.stack((action_balls[:, None], 0 * ball_pred, T.zeros_like(ball_pred)), dim=-1),
                        scene + T.stack((action_balls[:, None], 0 * ball_pred, T.zeros_like(ball_pred)), dim=-1),
                        T.stack((action_balls[:, None] + background + action_paths[:, None],
                                 target_paths[:, None] + init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)),
                        dim=1).detach()
                    diff_batch_right = T.cat((
                        0 * scene,
                        scene + T.stack((base_pred * 0, base_pred, 0 * base_paths[:, None]), dim=-1),
                        scene + T.stack((0 * target_paths[:, None], target_pred, 0 * target_pred), dim=-1),
                        scene + T.stack((action_pred, 0 * action_paths[:, None], 0 * action_paths[:, None]), dim=-1),
                        scene + T.stack((ball_pred, 0 * action_balls[:, None], T.zeros_like(ball_pred)), dim=-1),
                        T.stack((drawings + ball_pred + background, init_scenes[:, None, 0] + background,
                                 init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1),
                        T.stack((
                            drawings + background + action_pred, target_pred + init_scenes[:, None, 0] + background,
                            init_scenes[:, None, 1] + init_scenes[:, None, 2] + background), dim=-1)),
                        dim=1).detach()

                    diff_batch_left = self.cut_off(diff_batch_left.cpu())
                    white = T.ones_like(diff_batch_left)
                    white[:, :, :, :, [0, 1]] -= diff_batch_left[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch_left[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch_left[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch_left = white

                    diff_batch_right = self.cut_off(diff_batch_right.cpu())
                    white = T.ones_like(diff_batch_right)
                    white[:, :, :, :, [0, 1]] -= diff_batch_right[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [0, 2]] -= diff_batch_right[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                    white[:, :, :, :, [1, 2]] -= diff_batch_right[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                    diff_batch_right = white

                    text_right = ['',
                                  'predicted\nbase path',
                                  'predicted\ntarget path',
                                  'predicted\naction path',
                                  'predicted\naction ball\nposition',
                                  'sampled\naction ball',
                                  'combined\npredictions']

                    text_left = ['initial\nscene',
                                 'ground truth\nbase path',
                                 'ground truth\ntarget\npath',
                                 'ground truth\naction\npath',
                                 'ground truth\ninitial action\nball position',
                                 'ground truth\ninitial action\nball position',
                                 'all GT\npaths']

                    pbl = self.cut_off(diff_batch_left.cpu())
                    pbr = self.cut_off(diff_batch_right.cpu())
                    for j in range(pbl.shape[0]):
                        lpb = (pbl[j])[:, None]
                        rpb = (pbr[j])[:, None]
                        tmp_pb = T.cat((lpb, rpb), dim=1)
                        vis_batch(tmp_pb, f'result/flownet/inspect/{self.path}/{self.run}/{eval_setup}_fold_{fold}',
                                  f'visual_{i}_{j}', rows=text_left, descr=text_right)
        with open(f'result/flownet/inspect/{self.path}/{self.run}/{eval_setup}_fold_{fold}/loss.txt') as fp:
            fp.write(f"avg-loss {sum(log) / len(log)}")
            fp.write(f"avg-ten-smallest-losses {sum(sorted(log)[:10]) / 10}")

    def inspect_brute_search(self, eval_setup, fold, train_mode='CONS', epochs=1):
        self.to_train()
        data_loader = self.test_dataloader
        fail_loader = self.failed_test_dataloader
        base_net = self.models["base_net"]
        act_net = self.models["act_net"]
        tar_net = self.models["tar_net"]
        ext_net = self.models["ext_net"]

        percentage = []
        precision = []
        recall = []

        for epoch in range(epochs):
            for i, ((X,), (Z,)) in enumerate(zip(data_loader, fail_loader)):
                last_index = min(X.shape[0] // 4, Z.shape[0] // 4)
                X, Z = X[:last_index].to(self.device), Z[:last_index].to(self.device)
                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)

                # Prepare Data
                solve_scenes = X[:, :6]
                solve_base_paths = X[:, 6]
                solve_target_paths = X[:, 7]
                solve_action_paths = X[:, 9]
                solve_goal_paths = X[:, 8]
                fail_scenes = Z[:, [0, 1, 2, 3, 4, 5]]
                fail_base_paths = Z[:, 6]
                fail_target_paths = Z[:, 7]
                fail_action_paths = Z[:, 9]
                fail_goal_paths = Z[:, 8]

                init_scenes = T.cat((solve_scenes, fail_scenes), dim=0)
                target_paths = T.cat((solve_target_paths, fail_target_paths), dim=0)
                base_paths = T.cat((solve_base_paths, fail_base_paths), dim=0)
                action_paths = T.cat((solve_action_paths, fail_action_paths), dim=0)
                goal_paths = T.cat((solve_goal_paths, fail_goal_paths), dim=0)

                # Optional visiualization of batch data
                # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
                # vis_batch(X, f'data/flownet', f'{epoch}_{i}')
                with T.no_grad():
                    if train_mode == 'MIX':
                        modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
                    else:
                        modus = train_mode

                    # Forward Pass
                    base_pred = base_net(init_scenes)
                    if modus == 'GT':
                        action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                        confidence = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
                    elif modus == 'CONS':
                        action_pred = act_net(T.cat((init_scenes, base_pred.detach()), dim=1))
                        target_pred = tar_net(T.cat((init_scenes, base_pred.detach(), action_pred.detach()), dim=1))
                        confidence = ext_net(
                            T.cat((init_scenes, base_pred.detach(), target_pred.detach(), action_pred.detach()), dim=1))
                    elif modus == 'COMB':
                        action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                        target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                        confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))
                    elif modus == 'END':
                        action_pred = act_net(T.cat((init_scenes, base_pred), dim=1))
                        target_pred = tar_net(T.cat((init_scenes, base_pred, action_pred), dim=1))
                        confidence = ext_net(T.cat((init_scenes, base_pred, target_pred, action_pred), dim=1))

                # EVAL PERFORMANCE:
                conf_target = T.cat((T.ones(X.shape[0]), T.zeros(Z.shape[0])), dim=0).to(self.device)
                conf_pred = confidence[:, 0].round()
                percentage.append((conf_pred == conf_target).float().sum().item() / confidence.shape[0])
                recall.append(conf_pred[conf_target == 1].sum() / conf_target.sum())
                precision.append(conf_target[conf_pred == 1].sum() / conf_pred.sum())
                rows = [str(num) for num in range(init_scenes.shape[0])]

                print("correct classified percentage:", percentage[-1], end='\r')

                os.makedirs(f'result/flownet/training/{self.path}/{self.run}', exist_ok=True)
                print_batch = T.cat((init_scenes, goal_paths[:, None], base_paths[:, None], base_pred,
                                     target_paths[:, None], target_pred,
                                     action_paths[:, None], action_pred,
                                     T.ones_like(base_pred) * confidence[:, :, None, None],
                                     T.ones_like(base_pred) * conf_target[:, None, None, None]), dim=1).detach()
                text = [
                    'GT\nred ball'
                    'GT\ngreen ball',
                    'GT\nblue dynamic\nobject',
                    'GT\nblue static\nobject',
                    'GT\ngrey dynamic\nobjects',
                    'GT\nblack static\nobjects',
                    'GT\nblue dynamic\npath',
                    'GT\ngreen ball\nbase path',
                    'predicted\ngreen ball\nbase path',
                    'GT\ngreen ball\ntarget path',
                    'predicted\ngreen ball\ntarget path',
                    'GT\nred ball\naction path',
                    'predicted\nred ball\naction path',
                    'predicted\nconfidence',
                    'GT\nconfidence']
                # text = ['red', 'green', 'blue', 'blue', 'grey', 'black', 'base', 'target', 'goal\nnot used', 'action', 'base', 'target', 'action', 'conf']
                vis_batch(print_batch.cpu(),
                          f'result/flownet/inspect/{self.path}/{self.run}/brute_{eval_setup}_fold_{fold}',
                          f'poch_{epoch}_{i}', text=text, rows=rows)

                background = init_scenes[:, 4:].sum(dim=1)[:, None]
                background = background / max(background.max(), 1)
                tmp_conf = T.ones_like(base_pred) * confidence[:, :, None, None]
                real_conf = T.ones_like(base_pred) * conf_target[:, None, None, None]
                scene = T.stack((init_scenes[:, None, 0] + background, init_scenes[:, None, 1] + background,
                                 init_scenes[:, None, 2] + init_scenes[:, None, 3] + background), dim=-1)
                diff_batch = T.cat((
                    scene,
                    0.5 * T.stack((base_pred, base_paths[:, None], base_pred + base_paths[:, None]), dim=-1),
                    0.5 * T.stack((target_pred, target_paths[:, None], target_pred + target_paths[:, None]), dim=-1),
                    0.5 * T.stack((action_pred, action_paths[:, None], action_pred + action_paths[:, None]), dim=-1),
                    scene + T.stack((action_pred, target_pred, T.zeros_like(action_pred)), dim=-1),
                    scene + T.stack((action_paths[:, None], target_paths[:, None], T.zeros_like(action_pred)), dim=-1),
                    T.stack((1 - tmp_conf, 1 - tmp_conf, 1 - tmp_conf), dim=-1),
                    T.stack((1 - real_conf, 1 - real_conf, 1 - real_conf), dim=-1)),
                    dim=1).detach()
                diff_batch = self.cut_off(diff_batch.cpu())
                white = T.ones_like(diff_batch)
                white[:, :, :, :, [0, 1]] -= diff_batch[:, :, :, :, None, 2].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [0, 2]] -= diff_batch[:, :, :, :, None, 1].repeat(1, 1, 1, 1, 2)
                white[:, :, :, :, [1, 2]] -= diff_batch[:, :, :, :, None, 0].repeat(1, 1, 1, 1, 2)
                diff_batch = white
                text = [
                    'initial\nscene',
                    'base paths\nGT: blue-green\npredict: purple',
                    'target paths\nGT: blue-green\npredict: purple',
                    'action paths\nGT: blue-green\npredict: purple',
                    'predicted\nsimulated\nscene',
                    'GT\nsimulated\nscene',
                    'predicted\nconfidence',
                    'GT\nconfidence'
                ]
                vis_batch(self.cut_off(diff_batch.cpu()),
                          f'result/flownet/inspect/{self.path}/{self.run}/brute_{eval_setup}_fold_{fold}',
                          f'poch_{epoch}_{i}_diff', text=text, rows=rows)

        print()
        accuracy = sum(percentage) / len(percentage)
        recall = sum(recall) / len(recall)
        precision = sum(precision) / len(precision)
        print("AVERAGES:  accuracy:", accuracy, "recall", recall, "precision", precision)
        print(percentage)
        os.makedirs(f'result/solver/result/{self.path}/{self.run}', exist_ok=True)
        with open(f'result/solver/result/{self.path}/{self.run}/classification_{eval_setup}_fold_{fold}.txt',
                  'a') as fp:
            fp.write(f"\naccuracy {accuracy}\nrecall {recall}\nprecision {precision}")

    def to_eval(self):
        for model in self.models:
            self.models[model].eval()
            self.models[model].cpu()

    def to_train(self):
        for model in self.models:
            if self.device == "cuda":
                self.models[model].cuda()
                self.models[model].train()

        # avoiding target_net1 from training
        # self.models["tar_net1"].eval()

    def load_models(self, setup="ball_within_template", fold=0, device='cpu', no_second_stage=False):
        setup_name = "within" if setup == 'ball_within_template' else (
            "cross" if setup == 'ball_cross_template' else "custom")

        load_path = f"saves/flownet/{self.path}_{setup_name}_{fold}"
        for model in self.models:
            if no_second_stage and (model in ["sim_net", "comb_net", "success_net"]):
                continue
            print("loading:", load_path + f'/{model}.pt')
            try:
                self.models[model].load_state_dict(T.load(load_path + f'/{model}.pt', map_location=T.device(device)))
            except:
                continue

    def save_models(self, setup="ball_within_template", fold=0):
        setup_name = "within" if setup == 'ball_within_template' else (
            "cross" if setup == 'ball_cross_template' else "custom")
        save_path = f"saves/flownet/{setup}"
        os.makedirs(save_path, exist_ok=True)
        for model in self.models:
            print("saving:", save_path + f'/{model}.pt')
            T.save(self.models[model].state_dict(), save_path + f'/{model}.pt')

    def save_sfm_sequential(self, setup="ball_within_template", fold=0):
        setup_name = "within" if setup == 'ball_within_template' else (
            "cross" if setup == 'ball_cross_template' else "custom")
        save_path = f"saves/flownet/{setup}"
        os.makedirs(save_path, exist_ok=True)
        model = "SfM_sequential"
        print("saving:", save_path + f'/{model}.pt')
        T.save(self.models[model].state_dict(), save_path + f'/{model}.pt')

    def save_sfm1(self, setup="ball_within_template", fold=0):
        setup_name = "within" if setup == 'ball_within_template' else (
            "cross" if setup == 'ball_cross_template' else "custom")
        save_path = f"saves/flownet/{setup}"
        os.makedirs(save_path, exist_ok=True)
        model = "SfM1"
        print("saving:", save_path + f'/{model}.pt')
        T.save(self.models[model].state_dict(), save_path + f'/{model}.pt')

    def save_sfm2(self, setup="ball_within_template", fold=0):
        setup_name = "within" if setup == 'ball_within_template' else (
            "cross" if setup == 'ball_cross_template' else "custom")
        save_path = f"saves/flownet/{setup}"
        os.makedirs(save_path, exist_ok=True)
        model = "SfM2"
        print("saving:", save_path + f'/{model}.pt')
        T.save(self.models[model].state_dict(), save_path + f'/{model}.pt')


def neighbs(pos, shape, back, value):
    y, x = pos
    return [(k, l, value) for k in range(y - 1, y + 2) for l in range(x - 1, x + 2) if
            l >= 0 and k >= 0 and l < shape[1] and k < shape[0] and not ((k, l) in back)]


def calc_cost_map(map):
    front = []
    back = set()
    cost = T.zeros_like(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j]:
                cost[i, j] = 0
                back.add((i, j))
                front.extend(neighbs((i, j), map.shape, back, 0))
    while front:
        break


def dist_map(pic):
    # Calc Goal Pos
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y, x]:
                X += pic[y, x] * x
                Y += pic[y, x] * y
    summed = np.sum(pic)
    if summed == 0:
        return T.zeros(pic.shape[0], pic.shape[1])
    X /= summed
    Y /= summed

    # Calc Dist to Goal
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            pic[y, x] = ((y - Y) ** 2 + (x - X) ** 2) ** 0.5
    return pic / (1.41 * pic.shape[0])


# %%
# PARSER SETUP
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_id', type=str)
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-trans', action='store_true')
    parser.add_argument('-sequ', action='store_true')
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-gan', action='store_true')
    parser.add_argument('-linear', action='store_true')
    parser.add_argument('-pyramid', action='store_true')
    parser.add_argument('--train_mode', default='GT', type=str, choices=['GT', 'MIX', 'CONS', 'COMB', 'END'])
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--width', default=32, type=int)
    parser.add_argument('--visevery', default=10, type=int)
    args = parser.parse_args()
    print(args)
# %%
# DATA SETUP
if __name__ == "__main__":
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    if args.train:
        train_dataloader, index = make_mono_dataset(f"data/phyre_fold_{fold_id}_train_{args.width}",
                                                    size=(args.width, args.width), tasks=train_ids[:])
        os.makedirs(f'result/flownet/training/{args.path_id}', exist_ok=True)
        with open(f'result/flownet/training/{args.path_id}/namespace.txt', 'w') as handle:
            handle.write(str(args))
# %%
# NETS SETUP
if __name__ == "__main__":
    if args.linear:
        tar_net = FullyConnected(5, 1)
        base_net = FullyConnected(5, 1)
        act_net = FullyConnected(7, 1)
        ext_net = FullyConnected(7, 1)
    elif args.pyramid:
        tar_net = Pyramid(5, 1)
        base_net = Pyramid(5, 1)
        act_net = Pyramid(7, 1)
        ext_net = Pyramid(7, 1)
    else:
        tar_net = FlowNet(5, 16, sequ=args.sequ, trans=args.trans)
        base_net = FlowNet(5, 16, sequ=args.sequ, trans=args.trans)
        act_net = FlowNet(7, 16, sequ=args.sequ, trans=args.trans)
        ext_net = UpFlowNet(7, 16, sequ=args.sequ)
    discr = Discriminator(8)
# %%
# OPTI SETUP
if __name__ == "__main__" and args.train:
    # opti = T.optim.Adam(tar_net.parameters(recurse=True), lr=1e-3)
    # opti2 = T.optim.Adam(act_net.parameters(recurse=True), lr=1e-3)
    # opti3 = T.optim.Adam(ext_net.parameters(recurse=True), lr=1e-3)
    nets_opti = T.optim.Adam(chain(tar_net.parameters(recurse=True),
                                   act_net.parameters(recurse=True),
                                   ext_net.parameters(recurse=True),
                                   base_net.parameters(recurse=True)),
                             lr=3e-3)
    discr_opti = T.optim.Adam(discr.parameters(), lr=3e-3)


# %%
def train1():
    for e in range(100):
        print("epoch", e)
        for i, (X, Y) in enumerate(dataloader):
            Z = model(X)[:, 0]
            # print(Z)
            loss = F.binary_cross_entropy(Z, Y[:, 0])
            opti.zero_grad()
            loss.backward()
            opti.step()
            print(loss.item())
            if not i % 20:
                clear_output(wait=True)
                orig = T.cat(tuple(T.cat((sub, T.ones(32, 1) * 0.5), dim=1) for sub in X[0]), dim=1)
                print(orig.shape)
                plt.imshow(orig)
                plt.show()
                print(X.shape, Y.shape, Z.shape)
                plt.imshow(T.cat((X[0, 0], T.ones(32, 1), Y[0, 0].detach(), T.ones(32, 1), Z[0].detach()), dim=1))
                plt.show()


def train2():
    for e in range(100):
        print("epoch", e)
        for i, (X, Y) in enumerate(dataloader):
            with T.no_grad():
                Z = model(X)
            # print(Z)
            A = model2(T.cat((X[:, 1:], Y[:, None, 2], Z), dim=1))
            loss = F.binary_cross_entropy(A[:, 0], Y[:, 1])
            opti2.zero_grad()
            loss.backward()
            opti2.step()
            print(loss.item())
            if not i % 20:
                clear_output(wait=True)
                orig = T.cat(tuple(T.cat((sub, T.ones(32, 1) * 0.5), dim=1) for sub in X[0]), dim=1)
                print(orig.shape)
                plt.imshow(orig)
                plt.show()
                print(X.shape, Y.shape, Z.shape)
                plt.imshow(T.cat((Y[0, 0].detach(), T.ones(32, 1), Z[0, 0].detach(), T.ones(32, 1), Y[0, 2].detach(),
                                  T.ones(32, 1), Y[0, 1].detach(), T.ones(32, 1), A[0, 0].detach()), dim=1))
                plt.show()


def train3():
    for e in range(100):
        print("epoch", e)
        for i, (X, Y) in enumerate(dataloader):
            with T.no_grad():
                Z = model(X)
                A = model2(T.cat((X[:, 1:], Y[:, None, 2], Z), dim=1))
            # B = model3(T.cat((X[:,1:], Y[:,None,2], Z, A), dim=1))
            B = extract_action(A)
            # loss = F.binary_cross_entropy(B[:,0], Y[:,3])
            # opti3.zero_grad()
            # loss.backward()
            # opti3.step()
            # print(loss.item())
            if not i % 10:
                clear_output(wait=True)
                orig = T.cat(tuple(T.cat((sub, T.ones(32, 1) * 0.5), dim=1) for sub in X[0]), dim=1)
                print(orig.shape)
                plt.imshow(orig)
                plt.show()
                print(X.shape, Y.shape, Z.shape)
                plt.imshow(T.cat((Y[0, 0].detach(), T.ones(32, 1), Z[0, 0].detach(), T.ones(32, 1), Y[0, 2].detach(),
                                  T.ones(32, 1), Y[0, 1].detach(), T.ones(32, 1), A[0, 0].detach(), T.ones(32, 1),
                                  Y[0, 3].detach(),
                                  T.ones(32, 1), B[0, 0].detach()), dim=1))
                plt.show()


def train_supervised(epochs: int, base_net: FlowNet, tar_net: FlowNet, act_net: FlowNet, ext_net: UpFlowNet,
                     data_loader: T.utils.data.DataLoader, opti: T.optim.Adam, train_mode='GT'):
    for epoch in range(epochs):
        for i, (X,) in enumerate(data_loader):
            # Prepare Data
            action_balls = X[:, 0]
            init_scenes = X[:, 1:6]
            base_paths = X[:, 6]
            target_paths = X[:, 7]
            goal_paths = X[:, 8]
            action_paths = X[:, 9]

            # Optional visiualization of batch data
            # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)
            # vis_batch(X, f'data/flownet', f'{epoch}_{i}')

            if train_mode == 'MIX':
                modus = random.choice(['GT', 'COMB', 'CONS', 'END'])
            else:
                modus = train_mode

            # Forward Pass
            target_pred = tar_net(init_scenes)
            base_pred = base_net(init_scenes)
            if modus == 'GT':
                action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
            elif modus == 'CONS':
                action_pred = act_net(T.cat((init_scenes, target_pred.detach(), base_pred.detach()), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred.detach(), action_pred.detach()), dim=1))
            elif modus == 'COMB':
                action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))
            elif modus == 'END':
                action_pred = act_net(T.cat((init_scenes, target_pred, base_pred), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

            if not i % 10:
                os.makedirs(f'result/flownet/training/{args.path_id}', exist_ok=True)
                print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
                vis_batch(print_batch, f'result/flownet/training/{args.path_id}', f'poch_{epoch}_{i}')
            # plt.show()

            # Loss
            tar_loss = F.binary_cross_entropy(target_pred, target_paths[:, None])
            act_loss = F.binary_cross_entropy(action_pred, action_paths[:, None])
            ball_loss = F.binary_cross_entropy(ball_pred, action_balls[:, None])
            base_loss = F.binary_cross_entropy(base_pred, base_paths[:, None])
            if modus == 'END':
                loss = ball_loss
            else:
                loss = ball_loss + tar_loss + act_loss + base_loss
            print(epoch, i, loss.item())

            # Backward Pass
            opti.zero_grad()
            loss.backward()
            opti.step()


def train_as_gan(epochs: int, tar_net: FlowNet, act_net: FlowNet, ext_net: UpFlowNet, discr: Discriminator,
                 data_loader: T.utils.data.DataLoader, nets_opti: T.optim.Optimizer, discr_opti: T.optim.Optimizer,
                 use_GT=True):
    switch = False
    for epoch in range(epochs):
        for i, (X,) in enumerate(data_loader):
            switch = not switch
            # Prepare Data
            init_scenes = X[:, 1:6]
            target_paths = X[:, 6]
            action_paths = X[:, 8]
            goal_paths = X[:, 7]
            base_paths = X[:, 9]
            action_balls = X[:, 0]
            # print(init_scenes.shape, target_paths.shape, action_paths.shape, base_paths.shape)

            # TRAIN GENERATORS
            # Forward Pass
            if switch:
                target_pred = tar_net(init_scenes)
                action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))
                if not i % 50:
                    os.makedirs(f'result/flownet/training/{args.path_id}', exist_ok=True)
                    print_batch = T.cat((X, target_pred, action_pred, ball_pred), dim=1).detach()
                    vis_batch(print_batch, f'result/flownet/training/{args.path_id}', f'poch_{epoch}_{i}')
                # plt.show()

                # Loss
                validity = discr(T.cat((init_scenes, target_pred, action_pred, ball_pred), dim=1))
                gen_loss = F.binary_cross_entropy_with_logits(validity, T.ones(X.shape[0], 1))

                # Backward Pass
                nets_opti.zero_grad()
                gen_loss.backward()
                nets_opti.step()
                print(i, "gen_loss:", gen_loss.item())

            # TRAIN DISCRIMINATOR
            else:
                target_pred = tar_net(init_scenes)
                action_pred = act_net(T.cat((init_scenes, target_paths[:, None], base_paths[:, None]), dim=1))
                ball_pred = ext_net(T.cat((init_scenes, target_paths[:, None], action_paths[:, None]), dim=1))

                # Loss
                validity = discr(T.cat((init_scenes, target_pred, action_pred, ball_pred), dim=1))
                fake_loss = F.binary_cross_entropy_with_logits(validity, T.zeros(X.shape[0], 1))
                validity = discr(
                    T.cat((init_scenes, target_paths[:, None], action_paths[:, None], action_balls[:, None]), dim=1))
                real_loss = F.binary_cross_entropy_with_logits(validity, T.ones(X.shape[0], 1))
                discr_loss = fake_loss + real_loss

                # Backward Pass
                discr_opti.zero_grad()
                discr_loss.backward()
                discr_opti.step()
                print(i, "discr_loss:", discr_loss.item() / 2)


# %%
if __name__ == "__main__" and args.train:
    # train()
    # model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
    # train2()
    # model2.load_state_dict(T.load("saves/imaginet2-c16-all.pt"))
    if args.gan:
        train_as_gan(args.epochs, tar_net, act_net, ext_net, discr, train_dataloader, nets_opti, discr_opti,
                     train_mode=args.train_mode)
    else:
        train_supervised(args.epochs, base_net, tar_net, act_net, ext_net, train_dataloader, nets_opti,
                         train_mode=args.train_mode)

# %%
if __name__ == "__main__" and args.train:
    os.makedirs(f"saves/flownet/", exist_ok=True)
    T.save(tar_net.state_dict(), f"saves/flownet/flownet_tar_{args.path_id}.pt")
    T.save(act_net.state_dict(), f"saves/flownet/flownet_act_{args.path_id}.pt")
    T.save(ext_net.state_dict(), f"saves/flownet/flownet_ext_{args.path_id}.pt")
    T.save(base_net.state_dict(), f"saves/flownet/flownet_base_{args.path_id}.pt")
    if args.gan:
        T.save(discr.state_dict(), f"saves/flownet_discr_{args.path_id}.pt")


# %%
def inspect(tar_net: FlowNet, act_net: FlowNet, ext_net: UpFlowNet,
            data_loader: T.utils.data.DataLoader):
    os.makedirs(f'result/flownet/testing/{args.path_id}', exist_ok=True)
    with open(f'result/flownet/testing/{args.path_id}/namespace.txt', 'w') as handle:
        handle.write(str(args))

    for i, (X,) in enumerate(data_loader):
        init_scenes = X[:, 1:6]
        target_paths = X[:, 6]
        action_paths = X[:, 8]
        goal_paths = X[:, 7]
        base_paths = X[:, 9]
        action_balls = X[:, 0]

        with T.no_grad():
            base_pred = base_net(init_scenes)
            target_pred = tar_net(init_scenes)
            action_pred = act_net(T.cat((init_scenes, target_pred, base_paths[:, None]), dim=1))
            ball_pred = ext_net(T.cat((init_scenes, target_pred, action_pred), dim=1))

        print_batch = T.cat((X, base_pred, target_pred, action_pred, ball_pred), dim=1).detach()
        vis_batch(print_batch, f'result/flownet/testing/{args.path_id}', 'batch_{i}')


if __name__ == "__main__" and args.test:
    fold_id = 0
    tar_net.load_state_dict(T.load(f"saves/flownet/flownet_tar_{args.path_id}.pt"))
    act_net.load_state_dict(T.load(f"saves/flownet/flownet_act_{args.path_id}.pt"))
    ext_net.load_state_dict(T.load(f"saves/flownet/flownet_ext_{args.path_id}.pt"))
    base_net.load_state_dict(T.load(f"saves/flownet/flownet_base_{args.path_id}.pt"))
    test_dataloader, index = make_mono_dataset(f"data/phyre_fold_{fold_id}_test_32", size=(32, 32), tasks=test_ids)
    inspect(tar_net, act_net, ext_net, test_dataloader)
# %%
