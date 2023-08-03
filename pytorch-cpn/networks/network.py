from .resnet import *
from .resnetlsbn import resnet50lsbn
import torch.nn as nn
import torch
from .globalNet import globalNet
from .globalNet import globalNet_bist
from .refineNet import refineNet

__all__ = ['CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out


class CPN_lsbn(nn.Module):
    def __init__(self, resnet_lsbn, output_shape, num_class, pretrained=True):
        super(CPN_lsbn, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet_lsbn = resnet_lsbn
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x, y):
        x0, x1, x2, x3, x4 = self.resnet_lsbn(x, y)
        res_out = [x4, x3, x2, x1]
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return x0, x1, x2, x3, x4, global_outs, refine_out

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def LSBN_CPN50(out_size, num_class, in_features=0, num_conditions=2, pretrained=True):
    res50 = resnet50lsbn(pretrained=pretrained, num_class=num_class, in_features=in_features, num_conditions=num_conditions)
    model = CPN_lsbn(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model