#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: basic_block.py
@time: 2021/12/16 20:34
'''
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34, resnet50
from opencood.models.spvcnn.lovasz_loss import lovasz_softmax

class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key, eps=1.0e-5):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels, eps=eps),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels, eps=eps),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels, eps=eps),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        # return output.replace_feature(F.leaky_relu(output.features.float() + identity.features.float(), 0.1))
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))


class ResNetFCN(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None, half_enable=False):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.half_enable = half_enable
        
        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        if half_enable:
            self.conv1 = self.conv1.half()
            self.bn1 = self.bn1.float()
            self.relu = self.relu.half()
            self.maxpool = self.maxpool.half()
            self.layer1 = self.layer1.float()
            self.layer2 = self.layer2.float()
            self.layer3 = self.layer3.float()
            self.layer4 = self.layer4.float()
            self.deconv_layer1 = self.deconv_layer1.half()
            self.deconv_layer2 = self.deconv_layer2.half()
            self.deconv_layer3 = self.deconv_layer3.half()
            self.deconv_layer4 = self.deconv_layer4.half()

    def forward(self, data_dict):
        x = data_dict['img']
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)
        if self.half_enable:
            x = x.half()
        # Encoder
        conv1_out = self.conv1(x).float()
        conv1_out = self.bn1(conv1_out).half()
        conv1_out = self.relu(conv1_out)
        layer1_out = self.layer1(self.maxpool(conv1_out).float())
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        if self.half_enable:
            layer1_out = layer1_out.half()
            layer2_out = layer2_out.half()
            layer3_out = layer3_out.half()
            layer4_out = layer4_out.half()
        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        data_dict['img_scale2'] = layer1_out.to(data_dict['img'])
        data_dict['img_scale4'] = layer2_out.to(data_dict['img'])
        data_dict['img_scale8'] = layer3_out.to(data_dict['img'])
        data_dict['img_scale16'] = layer4_out.to(data_dict['img'])

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict

class ResNet50FCN(ResNetFCN):
    def __init__(self, backbone="resnet50", pretrained=True, config=None, half_enable=False):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet50":
            net = resnet50(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.half_enable = half_enable
        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )

        self.deconv_layer2 = nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingNearest2d(scale_factor=4),
                )

        self.deconv_layer3 = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingNearest2d(scale_factor=4),
                )

        self.deconv_layer4 = nn.Sequential(# 7-7+6+1=7, 12-7+6+1=12
                    nn.Conv2d(2048, 512, kernel_size=1),
                    nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.ReLU(inplace=True),# (7-1)x2-2x1+1x5=12-2+5=15, (12-1)x2-2x1+1x3=24-2+3=24+1=25
                    nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=2, padding=1, dilation=1, output_padding=0),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingNearest2d(scale_factor=4),
                )
        if half_enable:
            self.conv1 = self.conv1.half()
            self.bn1 = self.bn1.float()
            self.relu = self.relu.half()
            self.maxpool = self.maxpool.half()
            self.layer1 = self.layer1.float()
            self.layer2 = self.layer2.float()
            self.layer3 = self.layer3.float()
            self.layer4 = self.layer4.float()
            self.deconv_layer1 = self.deconv_layer1.half()
            self.deconv_layer2 = self.deconv_layer2.half()
            self.deconv_layer3 = self.deconv_layer3.half()
            self.deconv_layer4 = self.deconv_layer4.half()

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)