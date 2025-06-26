"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import os
from typing import Optional, Union
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
from torch.cuda import empty_cache as empty_cuda_cache
from torch.backends.cudnn import benchmark as cudnn_benchmark
from torch.nn import DataParallel
from torch import load 

from utils import util


def load_craftnet_model(
    cuda: bool = False, weight_path: Optional[Union[str, Path]] = None
):
    # get craft net path
    if weight_path is None:
        home_path = str(Path.home())
        weight_path = Path(
            home_path, ".craft_text_detector", "weights", "craft_mlt_25k.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    # load craft net
    from .nn import CraftNet

    craft_net = CraftNet()  # initialize

    # arange device
    if cuda:
        craft_net.load_state_dict(
            util.copyStateDict(load(weight_path, weights_only=True))
        )

        craft_net = craft_net.cuda()
        craft_net = DataParallel(craft_net)
        cudnn_benchmark = False
    else:
        craft_net.load_state_dict(
            util.copyStateDict(
                load(weight_path, map_location="cpu", weights_only=True)
            )
        )
    craft_net.eval()
    return craft_net


def load_refinenet_model(
    cuda: bool = False, weight_path: Optional[Union[str, Path]] = None
):
    # get refine net path
    if weight_path is None:
        home_path = Path.home()
        weight_path = Path(
            home_path, ".craft_text_detector", "weights", "craft_refiner_CTW1500.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    # load refine net
    from .nn import RefineNet

    refine_net = RefineNet()  # initialize

    # arange device
    if cuda:
        refine_net.load_state_dict(
            util.copyStateDict(load(weight_path, weights_only=True))
        )

        refine_net = refine_net.cuda()
        refine_net = DataParallel(refine_net)
        cudnn_benchmark = False
    else:
        refine_net.load_state_dict(
            util.copyStateDict(
                load(weight_path, map_location="cpu", weights_only=True)
            )
        )
    refine_net.eval()
    return refine_net


# BaseNet

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class vgg16_bn(torch.nn.Module):
    def __init__(self, weights=True, freeze=True):
        super(vgg16_bn, self).__init__()
        vgg_pretrained_features = models.vgg16_bn(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        if not weights:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  

        if freeze:
            for param in self.slice1.parameters():  
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"]
        )
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
    
# CraftNet

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CraftNet(nn.Module):
    def __init__(self, weights=False, freeze=False):
        super(CraftNet, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(weights, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """Base network"""
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(
            y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(
            y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(
            y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature

# RefineNet

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        refine = torch.cat([y.permute(0, 3, 1, 2), upconv4], dim=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        # out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)


# ----------------------------------------------------------------

class Craft:
    def __init__(
        self,
        output_dir=None,
        rectify=True,
        export_extra=True,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=1280,
        refiner=True,
        crop_type="poly",
        weight_path_craft_net: Optional[str] = None,
        weight_path_refine_net: Optional[str] = None,
    ):
        
        self.craft_net = None
        self.refine_net = None
        self.output_dir = output_dir
        self.rectify = rectify
        self.export_extra = export_extra
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.long_size = long_size
        self.refiner = refiner
        self.crop_type = crop_type

        # load craftnet
        self.load_craftnet_model(weight_path_craft_net)
        if refiner:
            self.load_refinenet_model(weight_path_refine_net)

    def load_craftnet_model(self, weight_path: Optional[str] = None):
        self.craft_net = load_craftnet_model(self.cuda, weight_path=weight_path)

    def load_refinenet_model(self, weight_path: Optional[str] = None):
        self.refine_net = load_refinenet_model(self.cuda, weight_path=weight_path)

    def unload_craftnet_model(self):
        self.craft_net = None
        empty_cuda_cache()

    def unload_refinenet_model(self):
        self.refine_net = None
        empty_cuda_cache()

    def detect_text(self, image, image_path=None):
        
        if image_path is not None:
            print("Argument 'image_path' is deprecated, use 'image' instead.")
            image = image_path

        # perform prediction
        prediction_result = util.get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.cuda,
            long_size=self.long_size,
        )

        # arange regions
        if self.crop_type == "box":
            regions = prediction_result["boxes"]
        elif self.crop_type == "poly":
            regions = prediction_result["polys"]
        else:
            raise TypeError("crop_type can be only 'polys' or 'boxes'")

        prediction_result["text_crop_paths"] = []
        if self.output_dir is not None:
            if type(image) == str:
                file_name, file_ext = os.path.splitext(os.path.basename(image))
            else:
                file_name = "image"
            exported_file_paths = util.export_detected_regions(
                image=image,
                regions=regions,
                file_name=file_name,
                output_dir=self.output_dir,
                rectify=self.rectify,
            )
            prediction_result["text_crop_paths"] = exported_file_paths

            if self.export_extra:
                util.export_extra_results(
                    image=image,
                    regions=regions,
                    heatmaps=prediction_result["heatmaps"],
                    file_name=file_name,
                    output_dir=self.output_dir,
                )

        return prediction_result
