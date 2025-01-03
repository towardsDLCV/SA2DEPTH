import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .intern_image import InternImage
from .sade_layers import *


class SADEDepth(nn.Module):

    def __init__(self,
                 pretrained=None,
                 max_depth=10.,
                 min_depth=1e-3,
                 encoder='large',
                 hidden_dim=256,
                 out_dim=128):
        super(SADEDepth, self).__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth

        if encoder == 'large':
            in_channels = [160, 320, 640, 1280]
            channels = 160
            depths = [5, 5, 22, 5]
            groups = [10, 20, 40, 80]
            drop_path_rate = 0.1
            layer_scale = 1.0
            offset_scale = 2.0
            mlp_ratio = 4.0
            post_norm = True
            num_classes = 21841
            heads = [2, 4, 8, 16]
            dim_head = 80
        elif encoder == 'small':
            in_channels = [64, 128, 256, 512]
            channels = 64
            depths = [4, 4, 18, 4]
            groups = [4, 8, 16, 32]
            drop_path_rate = 0.1
            layer_scale = None
            offset_scale = 1.0
            mlp_ratio = 4.0
            post_norm = False
            num_classes = 1000
            heads = [1, 2, 4, 8]
            dim_head = 64
        else:
            raise Exception("encoder must be either 'large' or 'small'")

        backbone_cfg = dict(channels=channels,
                            depths=depths,
                            groups=groups,
                            drop_path_rate=drop_path_rate,
                            layer_scale=layer_scale,
                            offset_scale=offset_scale,
                            mlp_ratio=mlp_ratio,
                            post_norm=post_norm,
                            num_classes=num_classes)
        self.backbone = InternImage(**backbone_cfg)

        offset_ks = [9, 7, 5, 3]
        offset_strides = [8, 4, 2, 1]
        self.block0 = SADEBlock(dim=in_channels[3], offset_kernel=offset_ks[3], offset_stride=offset_strides[3],
                                heads=heads[3], dim_head=dim_head)
        self.block1 = SADEBlock(dim=in_channels[2], offset_kernel=offset_ks[2], offset_stride=offset_strides[2],
                                heads=heads[2], dim_head=dim_head)
        self.block2 = SADEBlock(dim=in_channels[1], offset_kernel=offset_ks[1], offset_stride=offset_strides[1],
                                heads=heads[1], dim_head=dim_head)
        self.block3 = SADEBlock(dim=in_channels[0], offset_kernel=offset_ks[0], offset_stride=offset_strides[0],
                                heads=heads[0], dim_head=dim_head)

        self.up0 = nn.Sequential(nn.Conv2d(in_channels[3], in_channels[3] * 2, 1, bias=False),
                                 LayerNormProxy(in_channels[3] * 2),
                                 nn.PixelShuffle(2))
        self.fusion0 = nn.Conv2d(in_channels=in_channels[2] * 2, out_channels=in_channels[2], kernel_size=1)

        self.up1 = nn.Sequential(nn.Conv2d(in_channels[2], in_channels[2] * 2, 1, bias=False),
                                 LayerNormProxy(in_channels[2] * 2),
                                 nn.PixelShuffle(2))
        self.fusion1 = nn.Conv2d(in_channels=in_channels[1] * 2, out_channels=in_channels[1], kernel_size=1)

        self.up2 = nn.Sequential(nn.Conv2d(in_channels[1], in_channels[1] * 2, 1, bias=False),
                                 LayerNormProxy(in_channels[1] * 2),
                                 nn.PixelShuffle(2))
        self.fusion2 = nn.Conv2d(in_channels=in_channels[0] * 2, out_channels=in_channels[0], kernel_size=1)

        self.MLBP = MLBP(in_channels=in_channels,
                         dim=hidden_dim,
                         out_dim=out_dim,
                         max_depth=max_depth,
                         min_depth=min_depth,
                         act_layer=nn.GELU,
                         drop=0.)
        self.MLSS = MLSS(channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)
        self.disp_head1 = DispHead(input_dim=in_channels[0])

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            del self.backbone.conv_head
            del self.backbone.head
            del self.backbone.avgpool
            return
        print(f'== Load encoder backbone from: {pretrained}')
        ckpt = torch.load(pretrained, map_location='cuda:0')
        self.backbone.load_state_dict(ckpt['model'])
        del self.backbone.conv_head
        del self.backbone.head
        del self.backbone.avgpool

    def forward(self, imgs):
        f4, f8, f16, f32 = self.backbone(imgs)
        out = []

        e32 = self.block0(f32)
        out.append(e32)
        u16 = self.fusion0(torch.cat([self.up0(e32), f16], dim=1))

        e16 = self.block1(u16)
        out.append(e16)
        u8 = self.fusion1(torch.cat([self.up1(e16), f8], dim=1))

        e8 = self.block2(u8)
        out.append(e8)
        u4 = self.fusion2(torch.cat([self.up2(e8), f4], dim=1))

        e4 = self.block3(u4)
        out.append(e4)

        centers = self.MLBP(out)
        e4 = self.MLSS(out)

        depth = self.disp_head1(e4, centers, 4)
        return depth


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, centers, scale):
        x = x.softmax(dim=1)
        x = torch.sum(x * centers, dim=1, keepdim=True)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """
    Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


