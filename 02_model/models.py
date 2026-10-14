"""
Model definitions.

  FloodSegFormer  proposed model: MiT (SegFormer) encoder for the static inputs,
                  multi-scale temporal encoder for the rainfall history,
                  spatial-rainfall cross-attention and an FPN decoder with
                  classification and depth heads.
  FloodUNet       U-Net benchmark with a single-scale rainfall summary.
  RainfallCNN1D   1D-CNN benchmark (Woo et al., 2025, https://doi.org/10.3390/w17081239):
                  rainfall only, one output per valid grid cell.

The two patch models share the interface
  encode_static(static)            -> list of feature maps
  decode_with_rain(features, rain) -> (flood probability, depth)
so the static encoder can be run once per event at inference.

Shapes: static (B, 2, H, W) = [normalised DEM, land use]; rain (B, T, 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SegFormer (MiT) encoder
# ============================================================================
class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatial reduction of keys and values."""

    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = self.sr(x.permute(0, 2, 1).reshape(B, C, H, W))
            x = self.sr_norm(x.reshape(B, C, -1).permute(0, 2, 1))
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).float().softmax(-1).to(q.dtype)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        x = self.fc1(x).transpose(1, 2).view(B, -1, H, W)
        x = F.gelu(self.dwconv(x)).flatten(2).transpose(1, 2)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        return x + self.mlp(self.norm2(x), H, W)


class MiTEncoder(nn.Module):
    """Four-stage hierarchical encoder; outputs at 1/2, 1/4, 1/8, 1/16."""

    def __init__(self, in_ch=2, embed_dims=(32, 64, 128, 256),
                 num_heads=(1, 2, 4, 4), depths=(2, 2, 2, 2), sr_ratios=(4, 2, 1, 1)):
        super().__init__()
        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(4):
            c_in = in_ch if i == 0 else embed_dims[i - 1]
            self.patch_embeds.append(OverlapPatchEmbedding(c_in, embed_dims[i]))
            self.stages.append(nn.ModuleList(
                [TransformerBlock(embed_dims[i], num_heads[i], sr_ratios[i])
                 for _ in range(depths[i])]))
            self.norms.append(nn.LayerNorm(embed_dims[i]))

    def forward(self, x):
        B = x.shape[0]
        features = []
        for embed, blocks, norm in zip(self.patch_embeds, self.stages, self.norms):
            x, H, W = embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            features.append(x)
        return features


class FPNDecoder(nn.Module):
    """Top-down feature pyramid; returns the finest (1/2) level."""

    def __init__(self, embed_dims=(32, 64, 128, 256), decoder_dim=256):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(d, decoder_dim, 1) for d in embed_dims])
        self.td_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
                          nn.BatchNorm2d(decoder_dim), nn.GELU())
            for _ in embed_dims[:-1]])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim), nn.GELU())

    def forward(self, features):
        lats = [lat(f) for lat, f in zip(self.laterals, features)]
        td = lats[-1]
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(td, size=lats[i].shape[2:], mode='bilinear', align_corners=False)
            td = self.td_convs[i](up + lats[i])
        return self.fuse_conv(td)


# ============================================================================
# Rainfall encoding and fusion
# ============================================================================
class MultiScaleTemporalEncoder(nn.Module):
    """Three dilated 1D-conv branches (dilation 1, 2, 4) + temporal self-attention."""

    def __init__(self, in_feat=6, out_dim=256, n_heads=4, dilations=(1, 2, 4)):
        super().__init__()
        mid = out_dim // 3
        widths = (mid, mid, out_dim - 2 * mid)

        def branch(cout, d):
            return nn.Sequential(
                nn.Conv1d(in_feat, cout, 3, padding=d, dilation=d),
                nn.BatchNorm1d(cout), nn.GELU(),
                nn.Conv1d(cout, cout, 3, padding=d, dilation=d),
                nn.BatchNorm1d(cout), nn.GELU())

        self.branch_short = branch(widths[0], dilations[0])
        self.branch_mid = branch(widths[1], dilations[1])
        self.branch_long = branch(widths[2], dilations[2])
        self.scale_fusion = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.GELU())
        self.temporal_attn = nn.MultiheadAttention(out_dim, n_heads, batch_first=True,
                                                   dropout=0.1)
        self.attn_norm = nn.LayerNorm(out_dim)
        self.attn_ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim))
        self.ffn_norm = nn.LayerNorm(out_dim)

    def forward(self, rain):                                  # (B, T, F)
        x = rain.permute(0, 2, 1)
        x = torch.cat([self.branch_short(x), self.branch_mid(x), self.branch_long(x)], 1)
        seq = self.scale_fusion(x).permute(0, 2, 1)           # (B, T, C)
        q = self.attn_norm(seq)
        seq = seq + self.temporal_attn(q, q, q)[0]
        return seq + self.attn_ffn(self.ffn_norm(seq))


class SpatialRainCrossAttention(nn.Module):
    """Each spatial location (query) attends over the rainfall time steps (key/value)."""

    def __init__(self, spatial_dim, rain_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = spatial_dim // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(spatial_dim, spatial_dim)
        self.k_proj = nn.Linear(rain_dim, spatial_dim)
        self.v_proj = nn.Linear(rain_dim, spatial_dim)
        self.out_proj = nn.Linear(spatial_dim, spatial_dim)
        self.norm_q = nn.LayerNorm(spatial_dim)
        self.norm_kv = nn.LayerNorm(spatial_dim)

    def forward(self, spatial, rain_seq):                    # (B, C, H, W), (B, T, C)
        B, C, H, W = spatial.shape
        T = rain_seq.size(1)
        q = self.norm_q(self.q_proj(spatial.flatten(2).permute(0, 2, 1)))
        k = self.norm_kv(self.k_proj(rain_seq))
        v = self.norm_kv(self.v_proj(rain_seq))

        def split(t, n):
            return t.reshape(B, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = split(q, H * W), split(k, T), split(v, T)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).float().softmax(-1).to(q.dtype)
        ctx = (attn @ v).permute(0, 2, 1, 3).reshape(B, H * W, C)
        return self.out_proj(ctx).permute(0, 2, 1).reshape(B, C, H, W)


# ============================================================================
# Proposed model
# ============================================================================
class FloodSegFormer(nn.Module):
    def __init__(self, in_ch=2, rain_feat_dim=6, base_dim=32, attn_heads=4):
        super().__init__()
        dims = (base_dim, base_dim * 2, base_dim * 4, base_dim * 8)
        bot = dims[-1]

        self.encoder = MiTEncoder(in_ch, dims, num_heads=(1, 2, 4, attn_heads))
        self.temporal = MultiScaleTemporalEncoder(rain_feat_dim, bot, attn_heads)
        self.cross_attn = SpatialRainCrossAttention(bot, bot, attn_heads)
        self.temporal_attn_pool = nn.Sequential(nn.Linear(bot, 1))
        self.bot_fuse = nn.Sequential(nn.Conv2d(bot * 2, bot, 1), nn.BatchNorm2d(bot), nn.GELU())
        self.decoder = FPNDecoder(dims, bot)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(bot, bot // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(bot // 2), nn.GELU())
        h = bot // 2
        self.head_cls = nn.Sequential(nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(h, 1, 1), nn.Sigmoid())
        self.head_reg = nn.Sequential(nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(h, 1, 1), nn.ReLU())

    def encode_static(self, static):
        return self.encoder(static)

    def decode_with_rain(self, features, rain):
        f4 = features[-1]
        B, C, Hb, Wb = f4.shape

        # global rainfall context: attention pooling over time, broadcast in space
        seq = self.temporal(rain)                                         # (B, T, C)
        w = F.softmax(self.temporal_attn_pool(seq).squeeze(-1), dim=-1)   # (B, T)
        pooled = (w.unsqueeze(-1) * seq).sum(1)
        bot = self.bot_fuse(torch.cat([f4, pooled.view(B, C, 1, 1).expand(-1, -1, Hb, Wb)], 1))

        # location-specific rainfall context
        bot = bot + self.cross_attn(bot, seq)

        x = self.upsample(self.decoder(list(features[:-1]) + [bot]))
        return self.head_cls(x), self.head_reg(x)

    def forward(self, static, rain):
        return self.decode_with_rain(self.encode_static(static), rain)


# ============================================================================
# U-Net benchmark
# ============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.0),            # kept so that checkpoint keys match
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class SingleScaleRainEncoder(nn.Module):
    """Two 1D convolutions + average pooling -> one rainfall vector."""

    def __init__(self, in_feat=6, out_dim=256):
        super().__init__()
        hidden = max(out_dim // 2, 32)
        self.conv = nn.Sequential(
            nn.Conv1d(in_feat, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU())

    def forward(self, rain):
        return self.proj(self.pool(self.conv(rain.permute(0, 2, 1))).squeeze(-1))


def _dual_heads(ch):
    cls = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(0.0),
                        nn.Conv2d(ch, 1, 1), nn.Sigmoid())
    reg = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(0.0),
                        nn.Conv2d(ch, 1, 1), nn.ReLU())
    return cls, reg


class FloodUNet(nn.Module):
    def __init__(self, in_ch=2, rain_feat_dim=6, base_dim=32, **_):
        super().__init__()
        bd, bot = base_dim, base_dim * 8
        self.inc = DoubleConv(in_ch, bd)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bd, bd * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bd * 2, bd * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bd * 4, bot))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bot, bot))
        self.temporal = SingleScaleRainEncoder(rain_feat_dim, bot)
        self.bot_fuse = nn.Sequential(nn.Conv2d(bot * 2, bot, 1, bias=False),
                                      nn.BatchNorm2d(bot), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(bot, bd * 4, 4, 2, 1)
        self.cv1 = DoubleConv(bd * 4 + bot, bd * 4)
        self.up2 = nn.ConvTranspose2d(bd * 4, bd * 2, 4, 2, 1)
        self.cv2 = DoubleConv(bd * 2 + bd * 4, bd * 2)
        self.up3 = nn.ConvTranspose2d(bd * 2, bd, 4, 2, 1)
        self.cv3 = DoubleConv(bd + bd * 2, bd)
        self.up4 = nn.ConvTranspose2d(bd, bd, 4, 2, 1)
        self.cv4 = DoubleConv(bd * 2, bd)
        self.head_cls, self.head_reg = _dual_heads(bd)

    def encode_static(self, static):
        x1 = self.inc(static)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4, self.down4(x4)]

    def decode_with_rain(self, features, rain):
        x1, x2, x3, x4, x5 = features
        B, C = x5.shape[:2]
        s = self.temporal(rain).view(B, C, 1, 1).expand(-1, -1, *x5.shape[2:])
        x = self.bot_fuse(torch.cat([x5, s], 1))
        x = self.cv1(torch.cat([self.up1(x), x4], 1))
        x = self.cv2(torch.cat([self.up2(x), x3], 1))
        x = self.cv3(torch.cat([self.up3(x), x2], 1))
        x = self.cv4(torch.cat([self.up4(x), x1], 1))
        return self.head_cls(x), self.head_reg(x)

    def forward(self, static, rain):
        return self.decode_with_rain(self.encode_static(static), rain)


# ============================================================================
# 1D-CNN benchmark (rainfall only)
# ============================================================================
class RainfallCNN1D(nn.Module):
    """
    Two 1D convolutions (32 filters) + three fully connected layers (32, 256, 512)
    + a linear layer to every valid grid cell. Spatial patterns are stored in
    the last layer, so the model is tied to one grid.
    """

    def __init__(self, rain_feat_dim=6, seq_len=12, n_cells=124323, filters=32, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(rain_feat_dim, filters, 3, padding=1), nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 3, padding=1), nn.BatchNorm1d(filters), nn.ReLU(inplace=True))

        def fc(cin, cout):
            return [nn.Linear(cin, cout), nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.Dropout(dropout)]

        self.fc = nn.Sequential(*(fc(filters * seq_len, 32) + fc(32, 256) + fc(256, 512)))
        self.out = nn.Linear(512, n_cells)

    def forward(self, rain):                                   # (B, T, F) -> (B, n_cells)
        x = self.conv(rain.permute(0, 2, 1)).flatten(1)
        return F.relu(self.out(self.fc(x)))


def cells_to_map(values, cell_index, H, W):
    """(B, n_cells) -> (B, 1, H, W); cells outside cell_index are 0."""
    full = values.new_zeros(values.shape[0], H * W)
    full[:, cell_index] = values
    return full.view(-1, 1, H, W)


# ============================================================================
MODELS = {'segformer': FloodSegFormer, 'unet': FloodUNet, 'cnn1d': RainfallCNN1D}


def build_model(arch, **kwargs):
    return MODELS[arch](**kwargs)
