import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math


##### Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2 


##### Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


##### Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

# ####======原版FFM=========
# class FeatureFusionModule(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
#         self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
#         x1 = x1.flatten(2).transpose(1, 2)
#         x2 = x2.flatten(2).transpose(1, 2)
#         x1, x2 = self.cross(x1, x2) 
#         merge = torch.cat((x1, x2), dim=-1)
#         merge = self.channel_emb(merge, H, W)
        
#         return merge



# class FeatureFusionModule(nn.Module):
#     """
#     Cross-Modal Feature Fusion
#     流程：CrossPath → ChannelEmbed(2C→C) → PHA(C)
#     """
#     def __init__(self, dim, reduction=1, num_heads=None,
#                  norm_layer=nn.BatchNorm2d,           # 給 ChannelEmbed 用
#                  use_pha: bool = True):               # 如需關閉 PHA 可設 False

#         super().__init__()
#         # ① 雙向 Cross Attention（token level）
#         self.cross = CrossPath(dim=dim,
#                                reduction=reduction,
#                                num_heads=num_heads)

#         # ② Channel Embed：2C → C
#         self.channel_emb = ChannelEmbed(in_channels=dim * 2,
#                                         out_channels=dim,
#                                         reduction=reduction,
#                                         norm_layer=norm_layer)

#         # ③ PHA：局部語意精煉（可選）
#         self.pha = PHA(dim=dim, out=dim) if use_pha else nn.Identity()

#         self.apply(self._init_weights)

#     # --------------------------------------------------
#     @staticmethod
#     def _init_weights(m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     # --------------------------------------------------
#     def forward(self, x1, x2):
#         """
#         Args:  x1, x2 — (B, C, H, W)
#         Returns: fused feature — (B, C, H, W)
#         """
#         B, C, H, W = x1.shape

#         # --- token-level Cross Attention ---
#         x1 = x1.flatten(2).transpose(1, 2)          # (B, N, C)
#         x2 = x2.flatten(2).transpose(1, 2)
#         x1, x2 = self.cross(x1, x2)

#         # --- concat & channel embed ---
#         merge = torch.cat((x1, x2), dim=-1)         # (B, N, 2C)
#         merge = self.channel_emb(merge, H, W)       # (B, C, H, W)

#         # --- PHA 強化 ---
#         merge = self.pha(merge)                     # (B, C, H, W)

#         return merge
####Cross→PHA→ChannelEmbed

# class FeatureFusionModule(nn.Module):
#     """
#     Cross → PHA(2C) → ChannelEmbed
#     流程：CrossPath → PHA(局部語意精煉，作用於2C) → ChannelEmbed(2C→C)
#     """
#     def __init__(self, dim, reduction=1, num_heads=None,
#                  norm_layer=nn.BatchNorm2d,
#                  use_pha: bool = True):

#         super().__init__()

#         # ① 雙向 Cross Attention（token level）
#         self.cross = CrossPath(dim=dim,
#                                reduction=reduction,
#                                num_heads=num_heads)

#         # ② PHA 作用於 2C 通道
#         pha_in_channels = dim * 2
#         self.pha = PHA(dim=pha_in_channels,
#                        out=pha_in_channels) if use_pha else nn.Identity()

#         # ③ Channel Embed：2C → C
#         self.channel_emb = ChannelEmbed(in_channels=pha_in_channels,
#                                         out_channels=dim,
#                                         reduction=reduction,
#                                         norm_layer=norm_layer)

#         self.apply(self._init_weights)

#     # --------------------------------------------------
#     @staticmethod
#     def _init_weights(m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     # --------------------------------------------------
#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape

#         # --- Cross ---
#         x1 = x1.flatten(2).transpose(1, 2)          # (B, N, C)
#         x2 = x2.flatten(2).transpose(1, 2)
#         x1, x2 = self.cross(x1, x2)

#         # --- concat & reshape (B, 2C, H, W) ---
#         merge = torch.cat((x1, x2), dim=-1)         # (B, N, 2C)
#         merge = merge.transpose(1, 2).view(B, 2 * C, H, W)

#         # --- PHA 強化(2C) ---
#         merge = self.pha(merge)                     # (B, 2C, H, W)

#         # ★ 攤平回 (B, N, 2C) ---
#         merge = merge.flatten(2).transpose(1, 2)    # (B, N, 2C)

#         # ★ 傳入 ChannelEmbed 時附 H, W
#         merge = self.channel_emb(merge, H, W)       # (B, C, H, W)

#         return merge


        
#####PHA → Cross → ChannelEmbed
class FeatureFusionModule(nn.Module):
    """Branch-wise PHA → Cross → ChannelEmbed"""
    def __init__(self, dim, reduction=1, num_heads=None,
                 norm_layer=nn.BatchNorm2d, use_pha=True):
        super().__init__()
        self.use_pha = use_pha
        self.pha_rgb = PHA(dim, out=dim) if use_pha else nn.Identity()
        self.pha_x   = PHA(dim, out=dim) if use_pha else nn.Identity()
        self.cross   = CrossPath(dim, reduction, num_heads)
        self.channel_emb = ChannelEmbed(dim*2, dim, reduction, norm_layer)
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = self.pha_rgb(x1)
        x2 = self.pha_x(x2)
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        return merge

####======改良PHA0620========
##==========pha01==========



##### ========= ShiftViTBlockv2 =========
# class ShiftViTBlockv2(nn.Module):
#     def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
#         super().__init__()
#         self.dim = dim
#         self.norm2 = norm_layer(dim)
#         hidden_dim = int(dim * ratio)
#         self.channel = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, hidden_dim, 1),
#             act_layer(),
#             nn.Conv2d(hidden_dim, dim, 1),
#             nn.Sigmoid())
#         self.n_div = n_div

#     def forward(self, x):
#         B, C, H, W = x.shape
#         g = C // self.n_div
#         out = torch.zeros_like(x)
#         out[:, g*0:g*1, :, :-10] = x[:, g*0:g*1, :, 10:]
#         out[:, g*1:g*2, :, 10:] = x[:, g*1:g*2, :, :-10]
#         out[:, g*2:g*3, :-10, :] = x[:, g*2:g*3, 10:, :]
#         out[:, g*3:g*4, 10:, :] = x[:, g*3:g*4, :-10, :]
#         out[:, g*4:, :, :] = x[:, g*4:, :, :]
#         x = out
#         x = x + x * self.channel(self.norm2(x))
#         return x
##### =========改良ShiftViTBlockv2 =========
##==========shift01==========
# class ShiftViTBlockv2(nn.Module):
#     def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU,
#                  norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
#         super().__init__()
#         self.dim = dim
#         self.n_div = n_div
#         self.norm2 = norm_layer(dim)
#         hidden_dim = int(dim * ratio)
#         self.channel = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, hidden_dim, 1),
#             act_layer(),
#             nn.Conv2d(hidden_dim, dim, 1),
#             nn.Sigmoid()
#         )
#         self.shift_size = max(1, input_resolution[0] // 12)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         g = C // self.n_div
#         shift = max(1, W // 12)  # 動態 shift size（避免越界）
    
#         out = x.clone()  # 保底保留原值
    
#         if g * 4 <= C and shift < W and shift < H:
#             out[:, g*0:g*1, :, :] = torch.roll(x[:, g*0:g*1, :, :], shifts=-shift, dims=3)  # ← 左
#             out[:, g*1:g*2, :, :] = torch.roll(x[:, g*1:g*2, :, :], shifts=shift, dims=3)   # → 右
#             out[:, g*2:g*3, :, :] = torch.roll(x[:, g*2:g*3, :, :], shifts=-shift, dims=2)  # ↑ 上
#             out[:, g*3:g*4, :, :] = torch.roll(x[:, g*3:g*4, :, :], shifts=shift, dims=2)   # ↓ 下
#             # 其餘通道不動（保留原值）
    
#         # 通道注意力
#         x = out
#         x = x + x * self.channel(self.norm2(x))
#         return x
#####========終極debug版0620============
##==========shift02==========
class ShiftViTBlockv2(nn.Module):
    """
    Shift(±1 pixel) + SE re-weight；專為單卡小 batch 設定：
    BatchNorm2d(track_running_stats=False) 相當於一步一統計，
    不依賴 moving average，避免 batch=4 時數值抖動。
    """
    def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU):
        super().__init__()
        self.dim       = dim
        self.n_div     = max(1, n_div)
        self.shift_size = 1

        # 單卡小 batch → 關閉 running stats
        self.norm2 = nn.BatchNorm2d(dim, affine=True, track_running_stats=False)

        hidden_dim = max(1, dim // int(ratio))
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            act_layer(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        g = C // self.n_div
        if g == 0 or g * 4 > C or self.shift_size >= min(H, W):
            out = x                                   # 不做 shift
        else:
            s = self.shift_size
            out = x.clone()                           # ✅ 深拷貝避免 in-place
            out[:, 0*g:1*g] = torch.roll(x[:, 0*g:1*g], -s, 3)
            out[:, 1*g:2*g] = torch.roll(x[:, 1*g:2*g],  s, 3)
            out[:, 2*g:3*g] = torch.roll(x[:, 2*g:3*g], -s, 2)
            out[:, 3*g:4*g] = torch.roll(x[:, 3*g:4*g],  s, 2)

        x = self.norm2(out)
        x = x * self.channel(x)
        return x
######## ========= CoordAtt =========        
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h



   

# ###### ========= 原版PHA =========
# class PHA(nn.Module):
#     def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.coord_att = CoordAtt(inp=dim)
#         self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio, act_layer=act_layer, norm_layer=norm_layer, input_resolution=input_resolution)
#         self.norm2 = norm_layer(dim)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
#             act_layer(),
#             nn.Conv2d(int(dim * ratio), dim, kernel_size=1))
#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#     def forward(self, x):
#         x_norm = self.norm1(x)
#         coord_out = self.coord_att(x_norm)
#         shift_out = self.shift_vit(x_norm)
#         add1 = x + coord_out + shift_out
#         norm_out = self.norm2(add1)
#         mlp_out = self.mlp(norm_out)
#         add2 = add1 + mlp_out
#         return self.out(add2)
# class PHA(nn.Module):
#     """
#     P-H-A  (CoordAtt  +  ShiftViTBlockv2  +  channel-MLP)
#     • ShiftViTBlockv2 仍用 BatchNorm2d（單卡統計、track_running_stats=False）
#     • 其餘兩處 Norm 改為 GroupNorm(1,C)，不受 batch 大小影響
#     """
#     def __init__(self, dim, out=None, input_resolution=(64, 64),
#                  n_div=12, ratio=4., act_layer=nn.LeakyReLU):

#         super().__init__()
#         # ---------- GroupNorm(1,C) ----------
#         gn = lambda: nn.GroupNorm(1, dim)

#         self.norm1      = gn()
#         self.coord_att  = CoordAtt(inp=dim)

#         # ---------- Shift 窗口：仍用 BatchNorm2d ----------
#         self.shift_vit  = ShiftViTBlockv2(
#             dim=dim,
#             n_div=n_div,
#             ratio=ratio,
#             act_layer=act_layer
#         )

#         self.norm2      = gn()

#         hidden_dim      = max(4, int(dim * ratio))
#         self.mlp        = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 1, bias=True),
#             act_layer(inplace=True),
#             nn.Conv2d(hidden_dim, dim, 1, bias=True)
#         )

#         self.out        = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#     # -----------------------------------------------------

#     def forward(self, x):
#         # (1) 兩路注意力
#         x_norm   = self.norm1(x)
#         coord    = self.coord_att(x_norm)     # CoordAtt-BN 內部仍用自己的 BN
#         shifted  = self.shift_vit(x_norm)     # ShiftViTBlockv2-BN

#         y = x + coord + shifted               # 殘差融合

#         # (2) channel-MLP
#         y2 = self.mlp(self.norm2(y))
#         return self.out(y + y2)

class PHA(nn.Module):
    def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.coord_att = CoordAtt(inp=dim)
        self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio, act_layer=act_layer, norm_layer=norm_layer, input_resolution=input_resolution)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
            act_layer(),
            nn.Conv2d(int(dim * ratio), dim, kernel_size=1))
        self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

    def forward(self, x):
        x_norm = self.norm1(x)
        coord_out = self.coord_att(x_norm)
        shift_out = self.shift_vit(x_norm)
        add1 = x + coord_out + shift_out
        norm_out = self.norm2(add1)
        mlp_out = self.mlp(norm_out)
        add2 = add1 + mlp_out
        return self.out(add2)

# # ========= SHA =========
# class SHA(nn.Module):
#     def __init__(self, dim, out=None, n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
#         super(SHA, self).__init__()

#         self.norm1 = norm_layer(dim)

#         self.coord_att = CoordAtt(inp=dim)

#         self.shift_vit = ShiftViTBlockv2(
#             dim=dim,
#             n_div=n_div,
#             ratio=ratio,
#             act_layer=act_layer,
#             norm_layer=norm_layer
#         )

#         self.norm2 = norm_layer(dim)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
#             act_layer(),
#             nn.Conv2d(int(dim * ratio), dim, kernel_size=1)
#         )

#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()


#     def forward(self, x):
#         x_norm = self.norm1(x)
#         shift_out = self.shift_vit(x_norm)
#         add1 = shift_out + x

#         ca_out = self.coord_att(add1)
#         ca_norm_out = self.norm2(ca_out)
#         mlp_out = self.mlp(ca_norm_out)

#         add2 = add1 + mlp_out
#         output = self.out(add2)
#         return output    

class SHA(nn.Module):
    """
    Shift-based Hybrid Attention
    • Norm 採用 GroupNorm(1,C)
    • ShiftViTBlockv2 仍使用 BatchNorm2d（單卡、小 batch 友好）
    """
    def __init__(self, dim, out=None,
                 n_div: int = 12,
                 ratio: float = 4.,
                 act_layer: nn.Module = nn.LeakyReLU):

        super().__init__()

        gn = lambda: nn.GroupNorm(1, dim)          # Channel-wise LayerNorm

        # -------- Stage-1：Shift 注意力 --------
        self.norm1      = gn()
        self.shift_vit  = ShiftViTBlockv2(         # 內部 BN
            dim=dim,
            n_div=n_div,
            ratio=ratio,
            act_layer=act_layer
        )

        # -------- Stage-2：CoordAtt + MLP --------
        self.coord_att  = CoordAtt(inp=dim)
        self.norm2      = gn()

        hidden_dim      = max(4, int(dim * ratio))
        self.mlp        = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=True),
            act_layer(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=True)
        )

        self.out        = nn.Conv2d(dim, out, 1) if out else nn.Identity()

    # -----------------------------------------------------

    def forward(self, x):
        # Shift 分支 + 殘差
        y = x + self.shift_vit(self.norm1(x))

        # CoordAtt → GN → MLP
        y2 = self.mlp(self.norm2(self.coord_att(y)))

        return self.out(y + y2)

