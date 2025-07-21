# #目前使用crossattention01  PHFFM03 

#目前使用crossattention01  PHFFM01 


import torch
import torch.nn as nn
from typing import Callable, Optional

from timm.models.layers import trunc_normal_
import math


####=====原版Feature Rectify Module===========
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


# ######Stage 1原模型
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
#         super(CrossAttention, self).__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

#     def forward(self, x1, x2):
#         B, N, C = x1.shape
#         q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#         q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#         k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
#         k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

#         ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
#         ctx1 = ctx1.softmax(dim=-2)
#         ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
#         ctx2 = ctx2.softmax(dim=-2)

#         x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
#         x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
#         return x1, x2

# ###======原版crosspath=====    
# class CrossPath(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
#         self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
#         self.act1 = nn.ReLU(inplace=True)
#         self.act2 = nn.ReLU(inplace=True)
#         self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
#         self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
#         self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)

#     def forward(self, x1, x2):
#         y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
#         y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
#         v1, v2 = self.cross_attn(u1, u2)
#         y1 = torch.cat((y1, v1), dim=-1)
#         y2 = torch.cat((y2, v2), dim=-1)
#         out_x1 = self.norm1(x1 + self.end_proj1(y1))
#         out_x2 = self.norm2(x2 + self.end_proj2(y2))
#         return out_x1, out_x2    







# ##========= CrossAttention 01 =========
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.gate_mlp1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_mlp2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        #先V後K

    def forward(self, x1, x2):
        B, N, C = x1.shape

        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        gate1 = self.gate_mlp1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 正確
        gate2 = self.gate_mlp2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 正確

        ctx1 = (k1.transpose(-2, -1) @ (v1 * gate1)) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ (v2 * gate2)) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C)
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C)

        return x1, x2
class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm, asymmetric=False):
        super().__init__()
        self.asymmetric = asymmetric

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

        if not self.asymmetric:
            y2 = torch.cat((y2, v2), dim=-1)
            out_x2 = self.norm2(x2 + self.end_proj2(y2))
        else:
            out_x2 = self.norm2(v2)

        out_x1 = self.norm1(x1 + self.end_proj1(y1))

        return out_x1, out_x2
    
    
    




    



# Stage 2
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

###======原版FFM======
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

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
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        return merge
    
    
    
    # ========= CoordAtt =========
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

# ========= ShiftViTBlockv2 =========
class ShiftViTBlockv2(nn.Module):
    def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
        super().__init__()
        self.dim = dim
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * ratio)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            act_layer(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid())
        self.n_div = n_div

    def forward(self, x):
        B, C, H, W = x.shape
        g = C // self.n_div
        out = torch.zeros_like(x)
        out[:, g*0:g*1, :, :-10] = x[:, g*0:g*1, :, 10:]
        out[:, g*1:g*2, :, 10:] = x[:, g*1:g*2, :, :-10]
        out[:, g*2:g*3, :-10, :] = x[:, g*2:g*3, 10:, :]
        out[:, g*3:g*4, 10:, :] = x[:, g*3:g*4, :-10, :]
        out[:, g*4:, :, :] = x[:, g*4:, :, :]
        x = out
        x = x + x * self.channel(self.norm2(x))
        return x


#######===========論文版本===========
class PHA(nn.Module):
    r"""
    Parallel Hybrid Attention
      1. Shift-Window Attention 與 CoordAttention 並聯相加
      2. BatchNorm₁ → 主殘差 (x) → 得 y₁
      3. MLP → BatchNorm₂ → 加局部殘差 (y₁) → 得 y₂
    """
    def __init__(self, dim, out=None, input_resolution=(64, 64),
                 n_div=12, ratio=4., act_layer=nn.LeakyReLU,
                 norm_layer=nn.BatchNorm2d,        # 傳入 ShiftViTBlockv2
                 use_mlp=True):
        super().__init__()

        # ① 並聯注意力
        self.shift_vit = ShiftViTBlockv2(
            dim=dim, n_div=n_div, ratio=ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            input_resolution=input_resolution)
        self.coord_att = CoordAtt(inp=dim)

        # ② BatchNorm₁、③ BatchNorm₂
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

        # ③ MLP（在 BatchNorm₂ 之前）
        self.use_mlp = use_mlp
        if use_mlp:
            hidden = int(dim * ratio)
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
                act_layer(),
                nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
            )

        self.out = nn.Conv2d(dim, out, 1, bias=False) if out else nn.Identity()

    def forward(self, x):            # x: [B, C, H, W]
        # ① 並聯注意力融合
        fused = self.shift_vit(x) + self.coord_att(x)

        # ② BatchNorm₁ ＋ 主殘差
        y1 = self.bn1(fused) + x

        # ③ MLP → BatchNorm₂ ＋ 局部殘差
        y2_pre = self.mlp(y1) if self.use_mlp else y1
        y2 = self.bn2(y2_pre) + y1

        return self.out(y2)


#######===========GN版本===========        
# class PHA(nn.Module):
#     """Parallel Hybrid Attention（O3 版，第二殘差含 MLP → GroupNorm）

#     流程：
#         1. Shift‑Window Attention 與 CoordAttention 並聯相加
#         2. GroupNorm₁ → 主殘差 (x) → 得 y₁
#         3. MLP → GroupNorm₂ → 加局部殘差 (y₁) → 得 y₂
#         4. 1×1 Conv (可選) 調整輸出通道
#     """

#     # ------------------------ static helper ------------------------
#     @staticmethod
#     def _auto_groups(dim: int) -> int:
#         if dim >= 512:
#             return 32
#         if dim >= 320:
#             return 32
#         if dim >= 128:
#             return 16
#         return 8

#     # ------------------------- constructor ------------------------
#     def __init__(
#         self,
#         dim: int,
#         out: Optional[int] = None,
#         *,
#         input_resolution=(64, 64),
#         n_div: int = 12,
#         ratio: float = 4.0,
#         act_layer: Callable[[], nn.Module] = nn.LeakyReLU,
#         custom_norm: Optional[Callable[[int], nn.Module]] = None,
#     ) -> None:
#         super().__init__()

#         # norm factory
#         make_norm = custom_norm or (lambda c: nn.GroupNorm(PHA._auto_groups(c), c))

#         # Attention branches
#         self.shift_attn = ShiftViTBlockv2(
#             dim=dim,
#             n_div=n_div,
#             act_layer=act_layer,
#             norm_layer=lambda c: nn.GroupNorm(PHA._auto_groups(c), c),
#             input_resolution=input_resolution,
#         )
#         self.coord_attn = CoordAtt(inp=dim)

#         # Stage norms
#         self.gn1 = make_norm(dim)
#         self.gn2 = make_norm(dim)

#         # MLP before second GN
#         hidden = int(dim * ratio)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, hidden, 1, bias=False),
#             act_layer(),
#             nn.Conv2d(hidden, dim, 1, bias=False),
#         )

#         self.proj_out = nn.Conv2d(dim, out, 1, bias=False) if out is not None else nn.Identity()

#     # --------------------------- forward --------------------------
#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, H, W]
#         identity = x

#         # 1. 並聯注意力
#         attn_out = self.shift_attn(x) + self.coord_attn(x)

#         # 2. GroupNorm₁ + 主殘差
#         y1 = self.gn1(attn_out) + identity

#         # 3. MLP → GroupNorm₂ → 加局部殘差
#         mlp_out = self.mlp(y1)
#         gn2_out = self.gn2(mlp_out)
#         y2 = gn2_out + y1

#         return self.proj_out(y2)




    
    





    

######=====================pha03===========================    
class FeatureFusionPHA(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d,
                 input_resolution=(64, 64), use_pha_rgb=False, use_pha_x=False,
                 use_sha_rgb=False, use_sha_x=False):
        super().__init__()
        self.use_pha_rgb = use_pha_rgb
        self.use_pha_x = use_pha_x
        self.use_sha_rgb = use_sha_rgb
        self.use_sha_x = use_sha_x

        if self.use_pha_rgb:
            self.rgb_attn = PHA(dim=dim, out=dim, input_resolution=input_resolution)
        elif self.use_sha_rgb:
            self.rgb_attn = SHA(dim=dim, out=dim)
        else:
            self.rgb_attn = nn.Identity()

        if self.use_pha_x:
            self.x_attn = PHA(dim=dim, out=dim, input_resolution=input_resolution)
        elif self.use_sha_x:
            self.x_attn = SHA(dim=dim, out=dim)
        else:
            self.x_attn = nn.Identity()

        self.cross = CrossPath(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads,
            norm_layer=nn.LayerNorm
        )

        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer
        )

    def forward(self, x1, x2):  # x1, x2: [B, C, H, W]
        B, C, H, W = x1.shape

        # === Apply hybrid attention branch ===
        x1 = self.rgb_attn(x1)
        x2 = self.x_attn(x2)

        # === Flatten to sequence for CrossAttention ===
        x1_seq = x1.flatten(2).transpose(1, 2)  # [B, N, C]
        x2_seq = x2.flatten(2).transpose(1, 2)

        x1_out, x2_out = self.cross(x1_seq, x2_seq)  # [B, N, C], [B, N, C]

        # === Concat + ChannelEmbed ===
        fused = torch.cat([x1_out, x2_out], dim=-1)  # [B, N, 2C]
        out = self.channel_emb(fused, H, W)  # [B, C, H, W]

        return out



    

