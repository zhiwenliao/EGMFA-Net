import torch.nn as nn
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from module.WTConv import WTConv2d
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PPM(nn.Module):
    def __init__(self, pooling_sizes=(1, 3, 5)):
        super().__init__()
        self.layer = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(size,size)) for size in pooling_sizes])

    def forward(self, feat):
        b, c, h, w = feat.shape
        output = [layer(feat).view(b, c, -1) for layer in self.layer]
        output = torch.cat(output, dim=-1)
        return output


# Efficient self attention
class ESA_layer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.ppm = PPM(pooling_sizes=(1, 3, 5))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # input x (b, c, h, w)
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # q/k/v shape: (b, inner_dim, h, w)
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)   # q shape: (b, head, n_q, d)

        k, v = self.ppm(k), self.ppm(v)  # k/v shape: (b, inner_dim, n_kv)
        k = rearrange(k, 'b (head d) n -> b head n d', head=self.heads) # k shape: (b, head, n_kv, d)
        v = rearrange(v, 'b (head d) n -> b head n d', head=self.heads) # v shape: (b, head, n_kv, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)

        attn = self.attend(dots)

        out = torch.matmul(attn, v) # shape: (b, head, n_q, d)
        out = rearrange(out, 'b head n d -> b n (head d)')
        return self.to_out(out)


class ESA_blcok(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout = 0.):
        super().__init__()
        self.ESAlayer = ESA_layer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
       

    def forward(self, x):
        b, c, h, w = x.shape
        out = rearrange(x, 'b c h w -> b (h w) c')
        out = self.ESAlayer(x) + out
        out = self.ff(out) + out
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)

        return out


def MaskAveragePooling(x, mask):
    mask = torch.sigmoid(mask)
    b, c, h, w = x.shape
    eps = 0.0005
    x_mask = x * mask
    h, w = x.shape[2], x.shape[3]
    area = F.avg_pool2d(mask, (h, w)) * h * w + eps
    x_feat = F.avg_pool2d(x_mask, (h, w)) * h * w / area
    x_feat = x_feat.view(b, c, -1)
    return x_feat


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )


class KAEM_layer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.wtconv = WTConv2d(1,1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # inceptdw
        self.inceptdw = InceptionDWConv2d(dim)

    def forward(self, x, mask):
        x = self.inceptdw(x)
        mask = self.wtconv(mask)
        # input x (b, c, h, w)
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # q/k/v shape: (b, inner_dim, h, w)
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)  # q shape: (b, head, n_q, d)

        k, v = MaskAveragePooling(k, mask), MaskAveragePooling(v, mask)  # k/v shape: (b, inner_dim, 1)
        k = rearrange(k, 'b (head d) n -> b head n d', head=self.heads)  # k shape: (b, head, 1, d)
        v = rearrange(v, 'b (head d) n -> b head n d', head=self.heads)  # v shape: (b, head, 1, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)  # shape: (b, head, n_q, d)
        out = rearrange(out, 'b head n d -> b n (head d)')
        return self.to_out(out)



class KAEM_FFT(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.wtconv = WTConv2d(1,1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # inceptdw
        self.inceptdw = InceptionDWConv2d(dim)

    def forward(self, x, mask):
        x = self.inceptdw(x)
        mask = self.wtconv(mask)
        # input x (b, c, h, w)
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # q/k/v shape: (b, inner_dim, h, w)
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)  # q shape: (b, head, n_q, d)

        k, v = MaskAveragePooling(k, mask), MaskAveragePooling(v, mask)  # k/v shape: (b, inner_dim, 1)
        k = rearrange(k, 'b (head d) n -> b head n d', head=self.heads)  # k shape: (b, head, 1, d)
        v = rearrange(v, 'b (head d) n -> b head n d', head=self.heads)  # v shape: (b, head, 1, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)  # shape: (b, head, n_q, d)
        out = rearrange(out, 'b head n d -> b n (head d)')
        return self.to_out(out)



class KAEM(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout = 0.):
        super().__init__()
        self.KAEMlayer = KAEM_layer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))

    def forward(self, x, mask):
        b, c, h, w = x.shape
        out = rearrange(x, 'b c h w -> b (h w) c')
        out = self.KAEMlayer(x, mask) + out
        out = self.ff(out) + out
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)

        return out



# test
if __name__ == '__main__':
    x = torch.rand((4, 3, 320, 320))
    mask = torch.rand(4, 1, 320, 320)
    KAEM = KAEM(dim=3)
    esa = ESA_blcok(dim=3)
    ppm = PPM(pooling_sizes=(1, 3, 5))
    out = ppm(x)
    print(out.shape)
    # print(KAEM(x, mask).shape)
    # print(esa(x).shape)

