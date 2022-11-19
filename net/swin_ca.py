import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tools.point import sampling_points, point_sample


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.scale = 1 / np.sqrt(dim)

        self.pos = nn.Parameter(
            torch.zeros(1, window_size[0] * window_size[1], self.dim)
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qx, kvx):
        B_q, q_len, C = qx.shape
        B_kv, kv_len, _ = kvx.shape
        assert B_q % B_kv == 0, 'q batch must divided by kv batch'  # B_q % B_kv == 4
        tile_num = B_q // B_kv
        kvx = torch.tile(kvx, (tile_num, 1, 1))

        qx = qx + self.pos

        q = self.q_linear(qx)
        kv = self.kv_linear(kvx)
        k, v = torch.split(kv, [C, C], dim=2)

        q = q.reshape(B_q, q_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_q, kv_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_q, kv_len, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        attn = attn * self.scale

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_q, q_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 activation=nn.LeakyReLU):
        super(Conv2d, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterpolateUpsample(nn.Module):
    def __init__(self, num_features, mode='nearest'):
        """

        :param mode: bilinear nearest
        """
        super(InterpolateUpsample, self).__init__()

        self.mode = mode
        self.conv2d = Conv2d(num_features, num_features, 1, 1, 0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
        x = self.conv2d(x)
        return x


class TransposedConv2d(nn.Module):
    def __init__(self, num_features):
        super(TransposedConv2d, self).__init__()
        self.up = nn.ConvTranspose2d(num_features, num_features, 4, stride=2, bias=False)
        self.conv2d = Conv2d(num_features, num_features, 3, 1, 0)

    def forward(self, x):
        return self.conv2d(self.up(x))


class PixelShuffle(nn.Module):
    def __init__(self, num_features, scale=2):
        super(PixelShuffle, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=scale)
        self.conv2d = Conv2d(num_features // 4, num_features, 1, 1, 0)

    def forward(self, x):
        x = self.conv2d(self.up(x))
        return x


class ShallowFeatureLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, bias=False, activation=nn.LeakyReLU):
        super(ShallowFeatureLayer, self).__init__()
        self.conv2d = Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias, activation)

    def forward(self, x):
        return self.conv2d(x)


class PatchEmbed(nn.Module):
    def __init__(self, input_resolution, embed_dim=96):
        super(PatchEmbed, self).__init__()
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, input_resolution, embed_dim=96):
        super(PatchUnEmbed, self).__init__()
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim

    def forward(self, x):
        B = x.shape[0]
        x = x.transpose(1, 2).view(B, self.embed_dim, self.input_resolution[0], self.input_resolution[1])  # B C Ph Pw
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, input_resolution, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if self.shift_size > 0:
            kv_windows = window_partition(x, self.window_size * 2)
            kv_windows = kv_windows.view(-1, self.window_size * self.window_size * 4, C)
        else:
            kv_windows = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, kv_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class ChannelAttentionLayer(nn.Module):
    def __init__(self, input_resolution, dim):
        super(ChannelAttentionLayer, self).__init__()
        self.patch_unembed = PatchUnEmbed(input_resolution, dim)
        self.conv2d = Conv2d(dim, dim, 3, 1, 1)
        self.atten_conv2d = Conv2d(dim, dim, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.patch_embed = PatchEmbed(input_resolution, dim)

    def forward(self, x):
        x = self.patch_unembed(x)
        x_features = self.conv2d(x)
        atte_features = self.atten_conv2d(x)
        scores = torch.softmax(self.avg_pool(atte_features), dim=1)
        x = x_features * scores
        x = self.patch_embed(x)
        return x


class GateAttentionLayer(nn.Module):
    def __init__(self, input_resolution, dim):
        super(GateAttentionLayer, self).__init__()
        self.patch_unembed = PatchUnEmbed(input_resolution, dim)
        self.conv2d = Conv2d(dim, dim, 3, 1, 1)
        self.atten_conv2d = Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(input_resolution, dim)

    def forward(self, x):
        x = self.patch_unembed(x)
        x_features = self.conv2d(x)
        atten_features = self.atten_conv2d(x)
        scores = torch.sigmoid(atten_features)
        x = x_features * scores
        x = self.patch_embed(x)
        return x


class MixAttentionLayer(nn.Module):
    def __init__(self, input_resolution, dim, depth, num_head, window_size, mlp_ratio, drop, attn_drop, drop_path):
        super(MixAttentionLayer, self).__init__()

        self.sa = nn.ModuleList([
            SwinTransformerBlock(
                input_resolution=input_resolution,
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

        self.ca = ChannelAttentionLayer(input_resolution, dim)

        self.ga = GateAttentionLayer(input_resolution, dim)

        self.patch_unembed = PatchUnEmbed(input_resolution, dim * 3)
        self.conv2d = nn.Sequential(
            Conv2d(dim * 3, dim * 3, 3, 1, 1),
            Conv2d(dim * 3, dim, 1, 1, 0)
        )
        self.path_embed = PatchEmbed(input_resolution, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        shortcut = x
        ca_features = self.ca(x)
        ga_featuers = self.ga(x)
        sa_features = x
        for layer in self.sa:
            sa_features = layer(sa_features)
        x = torch.concat([sa_features, ca_features, ga_featuers], dim=2)
        x = self.patch_unembed(x)
        x = self.conv2d(x)
        x = self.path_embed(x)
        return shortcut + self.norm(x)


class BasicLayer(nn.Module):
    def __init__(self, input_resolution, dim, depth, num_head, window_size, mlp_ratio, drop, attn_drop, drop_path):
        super(BasicLayer, self).__init__()
        self.mix_atten_layer = MixAttentionLayer(input_resolution,
                                                 dim,
                                                 depth,
                                                 num_head,
                                                 window_size,
                                                 mlp_ratio=mlp_ratio,
                                                 drop=drop,
                                                 attn_drop=attn_drop,
                                                 drop_path=drop_path)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.norm(self.mix_atten_layer(x))


class PixHead(nn.Module):
    def __init__(self, num_features):
        super(PixHead, self).__init__()

        self.conv2d = nn.Sequential(
            Conv2d(num_features, num_features, 3, 1, 1),
            Conv2d(num_features, 3, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x


class PointRender(nn.Module):
    def __init__(self, scale=2, embed=96, num_classes=3, k=3, beta=0.75):
        super(PointRender, self).__init__()
        self.scale = scale
        self.k = k
        self.beta = beta

        self.mlp = nn.Conv1d(embed + num_classes, num_classes, 1, bias=False)

    def forward(self, x, res, out, training=True):
        if not training:
            return self.inference(x, res, out, training=training)

        B, C, H, W = out.shape

        points = sampling_points(torch.square(out - res), (H * W) // 16, self.k, self.beta, training)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(x, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation)
        rend = torch.relu(rend)
        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res, out, training=False):
        B, C, H, W = out.shape

        num_points = (H * W) // 16


        points_idx, points = sampling_points(torch.square(out - res), num_points, training=training)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(x, points, align_corners=False)
        # (B, 99, num_points)
        feature_representation = torch.cat([coarse, fine], dim=1)
        # (B, 3, num_points)
        rend = self.mlp(feature_representation)

        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        out = out.reshape(B, C, -1)
        out = out.scatter_(2, points_idx, rend).view(B, C, H, W)

        out = torch.relu(out)
        return {"fine": out}


class MixSwinCa(nn.Module):

    def __init__(self, img_size=32, in_channels=3, embed_dim=96, sf_layer=2, drop_rate=0.,
                 depths=[2, 4], num_heads=[3, 6], window_size=8, mlp_ratio=4.,
                 attn_drop_rate=0, drop_path_rate=0.1, scale=2, upsample='interpolate'):
        super(MixSwinCa, self).__init__()

        self.scale = scale

        # sf feature
        sf_list = []
        sf_in_channels = in_channels
        for i in range(sf_layer):
            if i != 0:
                sf_in_channels = embed_dim
            sf_list.append(ShallowFeatureLayer(sf_in_channels, embed_dim, 3, 1, 1))
        self.sf = nn.Sequential(*sf_list)

        # dp feature
        self.input_size = (img_size, img_size)
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(self.input_size, embed_dim)
        self.input_resolution = self.patch_embed.input_resolution
        self.num_features = self.patch_embed.embed_dim

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(input_resolution=self.input_resolution,
                               dim=self.num_features,
                               depth=depths[i_layer],
                               num_head=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])])
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)

        self.patch_unembed = PatchUnEmbed(self.input_resolution, self.num_features)
        if upsample == 'interpolate':
            self.upsample = InterpolateUpsample(num_features=self.num_features, mode='nearest')

        self.head = PixHead(num_features=self.num_features)

        self.point_render = PointRender()

        # recon image
        self.apply(self._init_weights)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname == 'BatchNorm2d':
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname == 'Linear':
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname == 'LayerNorm':
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif classname == 'MixConv2dLinear':
            if m.linear is not None:
                nn.init.normal_(m.linear.weight, 0, 0.01)
                if m.linear.bias is not None:
                    nn.init.normal_(m.linear.bias, 0, 0.01)

    def sf_features(self, x):
        return self.sf(x)

    def dp_features(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.patch_unembed(x)
        return x

    def recon(self, x):
        for i in range(0, self.scale, 2):
            x = self.upsample(x)
        return x

    def forward(self, x, training=True):
        # sf features
        x_sf = self.sf_features(x)
        # path embed
        x = self.patch_embed(x_sf)
        # dp features
        x = self.dp_features(x)
        dp_features = x
        x = self.recon(x)
        pred = self.head(x)

        coarse_out = self.head(dp_features)
        coarse_out = F.interpolate(coarse_out, scale_factor=self.scale)

        result = self.point_render(x_sf, coarse_out, pred, training)
        pred = torch.relu(pred)
        out = {'pred': pred}
        out.update(result)
        return out


if __name__ == '__main__':
    model = MixSwinCa()
    result = model(torch.ones(64, 3, 32, 32), training=False)
