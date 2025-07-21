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




####========= SESA Squeeze-enhanced Self-Attention =========
##==========sea01==========
# class SESA(nn.Module):
#     def __init__(self, dim, reduction=16):
#         super().__init__()
#         assert dim % reduction == 0, "dim 必須能整除 reduction"
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(dim // reduction, dim, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         context = self.pool(x)                      # B,C,1,1
#         context = self.fc1(context)
#         context = self.relu(context)
#         context = self.fc2(context)
#         attention = self.sigmoid(context)

#         # 改為平移後強度保留設計
#         return x * (1 + attention)  # 輕量型殘差（不改 shape）

##==========sea02==========        
# class SESA(nn.Module):
#     def __init__(self, dim, reduction=16, use_residual=True, scale=1.0):
#         super().__init__()
#         self.use_residual = use_residual
#         self.scale = scale
#         hidden_dim = max(1, dim // reduction)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         context = self.pool(x)
#         context = self.fc1(context)
#         context = self.relu(context)
#         context = self.fc2(context)
#         attention = self.sigmoid(context)

#         if self.use_residual:
#             return x * (1 + self.scale * attention)
#         else:
#             return x * attention

# class PHA(nn.Module):
#     def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.sesa_att = SESA(dim=dim)
#         self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio, act_layer=act_layer, norm_layer=norm_layer, input_resolution=input_resolution)
#         self.norm2 = norm_layer(dim)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
#             act_layer(),
#             nn.Conv2d(int(dim * ratio), dim, kernel_size=1))
#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#     def forward(self, x):
#         x_norm = self.norm1(x)
#         sesa_out = self.sesa_att(x_norm)
#         shift_out = self.shift_vit(x_norm)
#         add1 = x + sesa_out + shift_out
#         norm_out = self.norm2(add1)
#         mlp_out = self.mlp(norm_out)
#         add2 = add1 + mlp_out
#         return self.out(add2)
####======改良PHA0620========
##==========pha01==========
# class PHA(nn.Module):
#     def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4.,
#                  act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.sesa_att = SESA(dim=dim)
#         self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio,
#                                          act_layer=act_layer, norm_layer=norm_layer,
#                                          input_resolution=input_resolution)
#         self.norm2 = norm_layer(dim)
#         hidden_dim = int(dim * ratio)

#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 1),
#             act_layer(),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),  # DWConv 強化空間建模
#             act_layer(),
#             nn.Conv2d(hidden_dim, dim, 1)
#         )

#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#     def forward(self, x):
#         x_norm = self.norm1(x)
#         sesa_out = self.sesa_att(x_norm)
#         shift_out = self.shift_vit(x_norm)

#         # 控制融合比例，避免過度放大
#         fused = x + 0.5 * sesa_out + 0.5 * shift_out

#         fused = self.norm2(fused)
#         mlp_out = self.mlp(fused)
#         out = fused + mlp_out

#         return self.out(out)
#==========pha02==========
# class PHA(nn.Module):
#     def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4.,
#                  act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.sesa_att = SESA(dim=dim)
#         self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio,
#                                          act_layer=act_layer, norm_layer=norm_layer,
#                                          input_resolution=input_resolution)
#         self.norm2 = norm_layer(dim)
#         hidden_dim = int(dim * ratio)

#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 1),
#             act_layer(),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),  # DWConv 強化空間建模
#             act_layer(),
#             nn.Conv2d(hidden_dim, dim, 1)
#         )

#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#         # ✅ 加入可學融合參數（初始為 0.5）
#         self.alpha = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         x_norm = self.norm1(x)
#         sesa_out = self.sesa_att(x_norm)
#         shift_out = self.shift_vit(x_norm)

#         # ✅ 動態融合 SESA / ShiftViT 分支
#         fused = x + self.alpha * sesa_out + (1 - self.alpha) * shift_out

#         fused = self.norm2(fused)
#         mlp_out = self.mlp(fused)
#         out = fused + mlp_out

#         return self.out(out)


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
    def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU,
                 norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
        super().__init__()
        self.dim = dim
        self.n_div = n_div
        self.norm2 = norm_layer(dim)

        # ✅ 正確的 SE-style：通道壓縮 + Sigmoid，不加殘差
        hidden_dim = max(1, dim // int(ratio))
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            act_layer(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid()
        )

        self.shift_size = 1  # ✅ 每次只移動 1 pixel

    def forward(self, x):
        B, C, H, W = x.shape
        g = C // self.n_div
        shift = self.shift_size
        out = x.clone()

        if g * 4 <= C and shift < W and shift < H:
            # ✅ 用 torch.roll 環狀平移，避免 slicing 遺失資訊
            out[:, g*0:g*1, :, :] = torch.roll(x[:, g*0:g*1, :, :], shifts=-shift, dims=3)  # ← 左
            out[:, g*1:g*2, :, :] = torch.roll(x[:, g*1:g*2, :, :], shifts= shift, dims=3)  # → 右
            out[:, g*2:g*3, :, :] = torch.roll(x[:, g*2:g*3, :, :], shifts=-shift, dims=2)  # ↑ 上
            out[:, g*3:g*4, :, :] = torch.roll(x[:, g*3:g*4, :, :], shifts= shift, dims=2)  # ↓ 下
            # 其餘通道不移動

        x = self.norm2(out)
        attn = self.channel(x)  # 通道注意力 (B, C, 1, 1)
        x = x * attn            # ✅ 傳統 SE re-weight，無殘差

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
# class PHA(nn.Module):
#     def __init__(self, dim, out=None, input_resolution=(64, 64),
#                  n_div=12, ratio=4., act_layer=nn.LeakyReLU,
#                  norm_layer=None):
#         super().__init__()

#         assert norm_layer is not None, "PHA 模組需要外部傳入 norm_layer，例如 lambda c: nn.GroupNorm(32, c)"

#         self.norm1 = norm_layer(dim)
#         self.coord_att = CoordAtt(inp=dim)

#         self.shift_vit = ShiftViTBlockv2(
#             dim=dim,
#             n_div=n_div,
#             ratio=ratio,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             input_resolution=input_resolution
#         )

#         self.norm2 = norm_layer(dim)

#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
#             act_layer(),
#             nn.Conv2d(int(dim * ratio), dim, kernel_size=1)
#         )

#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#     def forward(self, x):
#         print("[PHA DEBUG] norm1 type:", type(self.norm1))

#         x_norm = self.norm1(x)
#         coord_out = self.coord_att(x_norm)
#         shift_out = self.shift_vit(x_norm)
#         add1 = x + coord_out + shift_out
#         norm_out = self.norm2(add1)
#         mlp_out = self.mlp(norm_out)
#         add2 = add1 + mlp_out
#         return self.out(add2)
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
    def __init__(self, dim, out=None, n_div=12, ratio=4., act_layer=nn.LeakyReLU,
                 num_groups=32):
        super(SHA, self).__init__()

        # 動態調整 group 數避免錯誤（GroupNorm 要求 C % G == 0）
        G = num_groups if dim % num_groups == 0 else 1
        norm_layer = lambda num_channels: nn.GroupNorm(G, num_channels)

        self.norm1 = norm_layer(dim)

        self.coord_att = CoordAtt(inp=dim)

        self.shift_vit = ShiftViTBlockv2(
            dim=dim,
            n_div=n_div,
            ratio=ratio,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
            act_layer(),
            nn.Conv2d(int(dim * ratio), dim, kernel_size=1)
        )

        self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

    def forward(self, x):
        x_norm = self.norm1(x)
        shift_out = self.shift_vit(x_norm)
        add1 = shift_out + x

        ca_out = self.coord_att(add1)
        ca_norm_out = self.norm2(ca_out)
        mlp_out = self.mlp(ca_norm_out)

        add2 = add1 + mlp_out
        output = self.out(add2)
        return output
# ######=======PHA在crossattention之後=========
# class FeatureFusionModule(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d,
#                  use_pha=True, input_resolution=(64, 64)):
#         super().__init__()
#         self.use_pha = use_pha
#         self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
#         self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        
#         if self.use_pha:
#             self.pha = PHA(dim=dim*2, out=dim*2, input_resolution=input_resolution, norm_layer=norm_layer)
#         else:
#             self.pha = nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
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

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
    
#         # Flatten + transpose
#         x1 = x1.flatten(2).transpose(1, 2)  # (B, H*W, C)
#         x2 = x2.flatten(2).transpose(1, 2)
    
#         # Cross-path attention
#         x1, x2 = self.cross(x1, x2)
    
#         # Concatenate and reshape back to (B, 2C, H, W)
#         merge = torch.cat((x1, x2), dim=-1)  # (B, H*W, 2C)
#         merge = merge.transpose(1, 2).reshape(B, -1, H, W)  # (B, 2C, H, W)
    
#         # PHA hybrid attention
#         merge = self.pha(merge)  # (B, 2C, H, W)
    
#         # Flatten again for ChannelEmbed
#         merge = merge.flatten(2).transpose(1, 2)  # (B, H*W, 2C)
    
#         # Channel embedding
#         merge = self.channel_emb(merge, H, W)  # output shape: (B, C, H, W)
    
#         return merge

#####=======PHA在crossattention之前=========
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d,
                 input_resolution=(64, 64),
                 use_pha_x1=True, use_pha_x2=True,
                 use_sha_x1=False, use_sha_x2=False):
        super().__init__()
        # === 單模態注意力 ===
        if use_pha_x1:
            self.attn_x1 = PHA(dim=dim, out=dim, input_resolution=input_resolution, norm_layer=norm_layer)
        elif use_sha_x1:
            self.attn_x1 = SHA(dim=dim, out=dim)
        else:
            self.attn_x1 = nn.Identity()

        if use_pha_x2:
            self.attn_x2 = PHA(dim=dim, out=dim, input_resolution=input_resolution, norm_layer=norm_layer)
        elif use_sha_x2:
            self.attn_x2 = SHA(dim=dim, out=dim)
        else:
            self.attn_x2 = nn.Identity()

        # === 雙模態交叉注意力 ===
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)

        # === 融合後通道壓縮 ===
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
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
        print(f"[FFM DEBUG] Forward input shape: {x1.shape}, {x2.shape}")

        # PHA 或 SHA 強化
        x1 = self.attn_x1(x1)  # (B, C, H, W)
        x2 = self.attn_x2(x2)  # (B, C, H, W)

        # Flatten for Cross Attention
        x1 = x1.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x2 = x2.flatten(2).transpose(1, 2)

        # Cross Attention
        x1, x2 = self.cross(x1, x2)

        # Concatenate
        merge = torch.cat((x1, x2), dim=-1)  # (B, H*W, 2C)
        merge = merge.transpose(1, 2).reshape(B, -1, H, W)  # (B, 2C, H, W)

        # Channel Embed
        merge = merge.flatten(2).transpose(1, 2)  # (B, H*W, 2C)
        merge = self.channel_emb(merge, H, W)  # (B, C, H, W)

        return merge

#####=======PHA在最後=========
# class FeatureFusionModule(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d,
#                  use_pha=True, input_resolution=(64, 64)):
#         super().__init__()
#         self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
#         self.channel_emb = ChannelEmbed(
#             in_channels=dim * 2,
#             out_channels=dim,
#             reduction=reduction,
#             norm_layer=norm_layer
#         )

#         self.use_pha = use_pha
#         if self.use_pha:
#             self.pha = PHA(dim=dim, out=dim, input_resolution=input_resolution, norm_layer=norm_layer)
#         else:
#             self.pha = nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
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

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape

#         # Flatten → Cross Attention
#         x1 = x1.flatten(2).transpose(1, 2)  # (B, N, C)
#         x2 = x2.flatten(2).transpose(1, 2)
#         x1, x2 = self.cross(x1, x2)

#         # Concatenate → Channel Embed
#         merge = torch.cat((x1, x2), dim=-1)  # (B, N, 2C)
#         merge = self.channel_emb(merge, H, W)  # (B, C, H, W)

#         # PHA 強化（融合後）
#         merge = self.pha(merge)  # (B, C, H, W)
#         return merge
