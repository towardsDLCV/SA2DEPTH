import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .SSM import SS2D


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


class Modulator(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super(Modulator, self).__init__()
        hidden = int(hidden_dim // 2)

        self.conv = nn.Sequential(nn.Conv2d(hidden_dim + input_dim, hidden * 2, 1),
                                  nn.PReLU(hidden * 2))

        self.convz = nn.Sequential(nn.Conv2d(hidden, hidden, (11, 11), padding=(5, 5), groups=hidden),
                                   nn.Conv2d(hidden, hidden_dim, 1))

        self.convr = nn.Sequential(nn.Conv2d(hidden, hidden, (11, 11), padding=(5, 5), groups=hidden),
                                   nn.Conv2d(hidden, hidden_dim, 1))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        U, M = self.conv(hx).chunk(2, dim=1)
        Z = torch.sigmoid(self.convz(U))
        R = torch.tanh(self.convr(M))

        h = Z * h + (1 - Z) * (R * x)
        return h


class TKFA(nn.Module):
    """
    Top-k Flexible Attention
    """

    def __init__(self,
                 dim,
                 offset_kernel,
                 offset_stride,
                 dim_head=80,
                 heads=2):
        super(TKFA, self).__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.proj_q = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.proj_kv = nn.Conv2d(dim, inner_dim * 2, kernel_size=1)

        self.offsetnet = nn.Sequential(
            nn.Conv2d(dim_head, dim_head, offset_kernel, offset_stride, offset_kernel // 2,
                      groups=dim_head),
            LayerNormProxy(dim_head),
            nn.SiLU(),
            nn.Conv2d(dim_head, 2, 1, 1, 0))

        self.softmax = nn.Softmax(dim=-1)
        self.modulator = Modulator(hidden_dim=inner_dim, input_dim=inner_dim)

        self.attn1 = nn.Parameter(torch.tensor([0.2], requires_grad=True))
        self.attn2 = nn.Parameter(torch.tensor([0.2], requires_grad=True))
        self.attn3 = nn.Parameter(torch.tensor([0.2], requires_grad=True))
        self.attn4 = nn.Parameter(torch.tensor([0.2], requires_grad=True))
        self.proj_out = nn.Conv2d(inner_dim, dim, 1)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.heads, -1, -1, -1)

        return ref

    def forward(self, x):
        shape = x.shape
        B, C, H, W, head = *shape, self.heads
        dtype, device = x.dtype, x.device

        query = self.proj_q(x)
        offset = rearrange(query, 'b (heads d) x y -> (b heads) d x y', heads=self.heads)
        offset = self.offsetnet(offset).contiguous()

        Ck, Hk, Wk = offset.shape[1:]
        N = Hk * Wk

        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        offset = rearrange(offset, 'b p h w -> b h w p')
        deform_point = (reference + offset).clamp(-1., +1.)

        sampled_x = F.grid_sample(input=x.reshape(B * self.heads, self.dim_head, H, W),
                                    grid=deform_point[..., (1, 0)],
                                    mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, Hk, Wk)
        key, value = self.proj_kv(sampled_x).chunk(2, dim=1)
        query, key, value = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=head), (query, key, value))

        mask1 = torch.zeros(B, head, H * W, Hk * Wk, device=device, requires_grad=False)
        mask2 = torch.zeros(B, head, H * W, Hk * Wk, device=device, requires_grad=False)
        mask3 = torch.zeros(B, head, H * W, Hk * Wk, device=device, requires_grad=False)
        mask4 = torch.zeros(B, head, H * W, Hk * Wk, device=device, requires_grad=False)

        dots = (query @ key.transpose(-1, -2)) * self.scale

        idx1 = torch.topk(dots, k=int(N * 1 / 10), dim=-1, largest=True)[1]
        mask1.scatter_(-1, idx1, 1.)
        dots1 = torch.where(mask1 > 0, dots, torch.full_like(dots, float('-inf')))

        idx2 = torch.topk(dots, k=int(N * 3 / 10), dim=-1, largest=True)[1][..., int(N / 10):]
        mask2.scatter_(-1, idx2, 1.)
        dots2 = torch.where(mask2 > 0, dots, torch.full_like(dots, float('-inf')))

        idx3 = torch.topk(dots, k=int(N * 5 / 10), dim=-1, largest=True)[1][..., int(N * 3 / 10):]
        mask3.scatter_(-1, idx3, 1.)
        dots3 = torch.where(mask3 > 0, dots, torch.full_like(dots, float('-inf')))

        idx4 = torch.topk(dots, k=int(N * 7 / 10), dim=-1, largest=True)[1][..., int(N * 5 / 10):]
        mask4.scatter_(-1, idx4, 1.)
        dots4 = torch.where(mask4 > 0, dots, torch.full_like(dots, float('-inf')))

        attn = self.softmax(dots)
        attn1 = self.softmax(dots1)
        attn2 = self.softmax(dots2)
        attn3 = self.softmax(dots3)
        attn4 = self.softmax(dots4)

        out = (attn @ value)
        out1 = (attn1 @ value)
        out2 = (attn2 @ value)
        out3 = (attn3 @ value)
        out4 = (attn4 @ value)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h=head, y=W)
        out1 = rearrange(out1, 'b h (x y) d -> b (h d) x y', h=head, y=W)
        out2 = rearrange(out2, 'b h (x y) d -> b (h d) x y', h=head, y=W)
        out3 = rearrange(out3, 'b h (x y) d -> b (h d) x y', h=head, y=W)
        out4 = rearrange(out4, 'b h (x y) d -> b (h d) x y', h=head, y=W)

        out1 = self.modulator(h=out, x=out1)
        out2 = self.modulator(h=out, x=out2)
        out3 = self.modulator(h=out, x=out3)
        out4 = self.modulator(h=out, x=out4)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        out = self.proj_out(out)
        return out


class SFU(nn.Module):
    """
    Selective Fusion Unit
    """
    def __init__(self,
                 channels: list,
                 dim):
        super(SFU, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.proj_conv1 = nn.Conv2d(channels[0], dim, kernel_size=1)
        self.proj_conv2 = nn.Conv2d(channels[1], dim, kernel_size=1)
        self.proj_conv3 = nn.Conv2d(channels[2], dim, kernel_size=1)
        self.proj_conv4 = nn.Conv2d(channels[3], dim, kernel_size=1)

        self.Z = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                               nn.Conv2d(dim, int(dim // 2), kernel_size=1),
                               nn.SiLU())

        self.V1 = nn.Conv2d(int(dim // 2), dim, kernel_size=1)
        self.V2 = nn.Conv2d(int(dim // 2), dim, kernel_size=1)
        self.V3 = nn.Conv2d(int(dim // 2), dim, kernel_size=1)
        self.V4 = nn.Conv2d(int(dim // 2), dim, kernel_size=1)

    def upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x_list):
        x32, x16, x8, x4 = x_list

        Hk, Wk = x4.shape[2:]

        s4 = self.proj_conv1(x4)
        s8 = self.upsample(self.proj_conv2(x8), size=(Hk, Wk))
        s16 = self.upsample(self.proj_conv3(x16), size=(Hk, Wk))
        s32 = self.upsample(self.proj_conv4(x32), size=(Hk, Wk))
        inp_feat = torch.stack([s4, s8, s16, s32], dim=1)
        S = s4 + s8 + s16 + s32
        Z = self.Z(S)

        V1 = self.V1(Z)
        V2 = self.V2(Z)
        V3 = self.V3(Z)
        V4 = self.V4(Z)

        attn_V = torch.stack([V1, V2, V3, V4], dim=1)
        attn_V = self.softmax(attn_V)
        feat_V = torch.sum(inp_feat * attn_V, dim=1)
        return feat_V


class GGFN(nn.Module):
    """
    Gated-GConv-Feed-forward
    """
    def __init__(self, dim, mult):
        super(GGFN, self).__init__()
        hidden_dim = int(dim * mult)
        self.proj_in = nn.Conv2d(dim, hidden_dim, kernel_size=1)

        self.group1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.group2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim//2)
        self.group3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim//4)
        self.tanh = nn.Tanh()
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.proj_in(x)
        g1, g2, g3 = self.group1(x), self.group2(x), self.group3(x)
        group = self.tanh(g1 + g2 + g3)
        x = x * group
        x = self.proj_out(x)
        return x


class MLSS(nn.Module):
    """
    Multi-level Selective Scanning
    """
    def __init__(self,
                 channels: list,
                 hidden_dim: int,
                 out_dim: int,
                 ):
        super(MLSS, self).__init__()
        self.SFU = SFU(channels=channels, dim=hidden_dim)

        self.norm0 = LayerNormProxy(dim=hidden_dim)
        self.mamba = SS2D(d_model=hidden_dim)

        self.norm1 = LayerNormProxy(dim=hidden_dim)
        self.GGFN = GGFN(dim=hidden_dim, mult=4)
        self.conv_out = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x_list):
        x = self.SFU(x_list)
        x = x + self.mamba(self.norm0(x).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x = x + self.GGFN(self.norm1(x))
        return self.conv_out(x)


class MLBP(nn.Module):

    def __init__(self,
                 in_channels=[192, 384, 768, 1536],
                 dim=512,
                 out_dim=256,
                 max_depth=10.,
                 min_depth=1e-3,
                 act_layer=nn.GELU,
                 drop=0.):
        super(MLBP, self).__init__()
        self.in_channels = in_channels
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

        self.proj_1 = nn.Conv2d(in_channels[3], dim, kernel_size=1)
        self.proj_2 = nn.Conv2d(in_channels[2], dim, kernel_size=1)
        self.proj_3 = nn.Conv2d(in_channels[1], dim, kernel_size=1)
        self.proj_4 = nn.Conv2d(in_channels[0], dim, kernel_size=1)

        self.norm = nn.LayerNorm(dim * len(in_channels))

        self.mlp = nn.Sequential(nn.Linear(dim * len(in_channels), dim),
                                 act_layer(),
                                 nn.Dropout(drop),
                                 nn.Linear(dim, out_dim),
                                 nn.Dropout(drop))

    def forward(self, feats):
        f32, f16, f8, f4 = self.avgpool(self.proj_1(feats[0])),\
                           self.avgpool(self.proj_2(feats[1])),\
                           self.avgpool(self.proj_3(feats[2])), \
                           self.avgpool(self.proj_4(feats[3]))
        out = torch.cat([f32, f16, f8, f4], dim=1)
        out = self.norm(out)
        out = self.mlp(out)
        bins = torch.softmax(out, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers


class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False

    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size // 2,
                        groups=self.dim)  # equal to x + conv(x)


class SADEBlock(nn.Module):

    def __init__(self,
                 dim=640,
                 offset_kernel=3,
                 offset_stride=1,
                 mult=4.,
                 heads=2,
                 dim_head=80):
        super(SADEBlock, self).__init__()
        self.pe = ResDWC(dim, 3)

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = TKFA(dim=dim,
                         offset_kernel=offset_kernel,
                         offset_stride=offset_stride,
                         heads=heads,
                         dim_head=dim_head)
        self.norm2 = nn.BatchNorm2d(dim)
        self.GGFN = GGFN(dim=dim,
                         mult=mult)

    def forward(self, x):
        x = self.pe(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.GGFN(self.norm2(x))
        return x