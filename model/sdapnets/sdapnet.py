import numpy as np
import pdb
import math
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda, DeformConv2d
from utils import limit_period
from model.backbones.backbone import build_backbone
from model.necks.rpn import RPN

class SelfMultiHeadAttention1D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        自注意力版1D多头注意力机制
        参数:
            embed_dim: 输入/输出特征维度 (必须能被num_heads整除)
            num_heads: 注意力头的数量
            dropout: 注意力权重dropout率
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义共享输入的线性变换层
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)  # 同时生成Q,K,V
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            padding_mask: 填充掩码 [batch_size, seq_len]
        返回:
            注意力输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()

        # 生成Q,K,V (来自同一输入)
        qkv = self.qkv_proj(x)  # [B, L, 3*E]
        q, k, v = qkv.chunk(3, dim=-1)  # 各[B, L, E]

        # 分头处理
        q = self._split_heads(q)  # [B, H, L, d_k]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores形状: [B, H, L, L]

        # 应用填充掩码（处理填充位置）
        if padding_mask is not None:
            mask = padding_mask.view(batch_size, 1, 1, seq_len)  # 广播维度
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context = torch.matmul(attn_weights, v)  # [B, H, L, d_k]

        # 合并注意力头
        context = self._merge_heads(context)  # [B, L, E]

        # 最终投影
        output = self.out_proj(context)
        return output

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将输入张量分头"""
        B, L, E = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """合并注意力头"""
        B, H, L, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * d_k)

class TransformerEncoderLayer(nn.Module):
    """完整Transformer编码层（含自注意力+前馈）"""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.self_attn = SelfMultiHeadAttention1D(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(32, 1, 1, bias=False)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力分支
        attn_output = self.self_attn(x, padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈分支
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        x = self.conv(x)
        return x.squeeze(dim=1)

# # 使用示例
# if __name__ == "__main__":
#     # 参数设置
#     batch_size = 2
#     seq_len = 10
#     embed_dim = 512
#     num_heads = 8
#
#     # 初始化模块
#     encoder_layer = TransformerEncoderLayer(embed_dim, num_heads)
#
#     # 模拟输入数据
#     x = torch.randn(batch_size, seq_len, embed_dim)
#     mask = torch.ones(batch_size, seq_len).bool()  # 真实数据掩码
#
#     # 前向传播
#     output = encoder_layer(x, mask)
#     print(f"输入形状: {x.shape}")
#     print(f"输出形状: {output.shape}")  # 应保持与输入相同形状
#
#     # 性能测试
#     with torch.no_grad():
#         torch.cuda.synchronize()
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         _ = encoder_layer(x.cuda(), mask.cuda())
#         end.record()
#         torch.cuda.synchronize()
#         print(f"GPU耗时: {start.elapsed_time(end):.2f}ms")

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return:
               pillars: (p1 + p2 + ... + pb, num_points, c),
               coors_batch: (p1 + p2 + ... + pb, 1 + 3),
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)
        return pillars, coors_batch, npoints_per_pillar

class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        #self.mha = MultiHeadAttention1D(embed_dim=32, num_heads=8)
        self.mha = TransformerEncoderLayer(embed_dim=64, num_heads=8)
        # self.globel_conv = nn.Sequential(
        #     nn.Linear(out_channel * 2, out_channel),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(out_channel, 1),
        # )
        # #self.fc = nn.Linear(out_channel*2, out_channel)
        self.dcn = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(),
            DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False, modulation=True),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.ReLU())
    # def efficient_euclidean(self, x):
    #     # 利用公式 ||a - b||² = ||a||² + ||b||² - 2<a, b>
    #     x_norm = (x ** 2).sum(dim=1)  # (M,)
    #     topk_values, topk_indices = (x_norm.unsqueeze(0) + x_norm.unsqueeze(1) - 2 * (x @ x.T)).topk(k=10, dim=1)
    #     #pairwise_dist = x_norm.unsqueeze(0) + x_norm.unsqueeze(1) - 2 * (x @ x.T)
    #     #return torch.sqrt(pairwise_dist.clamp(min=0))  # 防止数值误差
    #     return topk_values, topk_indices

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:,
                                                                                                   None,
                                                                                                   None]  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (
                    coors_batch[:, None, 1:2] * self.vx + self.x_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
                    coors_batch[:, None, 2:3] * self.vy + self.y_offset)  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center],
                             dim=-1)  # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        features_mul_att = self.mha(features.permute(0, 2, 1).contiguous())

        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        #global_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = features_mul_att[cur_coors_idx]
            #topk_values, topk_indices = self.efficient_euclidean(cur_features)
            #topk_values = F.softmax(topk_values,dim=-1)

            #fused = (cur_features[topk_indices] * topk_values.unsqueeze(-1)).sum(dim=1)
            #features = torch.cat([features_mul_att[cur_coors_idx], fused], dim=1)
            #features = self.fc(features)

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)
        batched_canvas = self.dcn(batched_canvas) + batched_canvas

        # 6. pillar scatter
        # batched_canvas = []
        # for i in range(bs):
        #     cur_coors_idx = coors_batch[:, 0] == i
        #     cur_coors = coors_batch[cur_coors_idx, :]
        #     cur_features = pooling_features[cur_coors_idx]
        #
        #     canvas = torch.zeros((self.x_l, self.y_l, self.out_channel+1), dtype=torch.float32, device=device)
        #     canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
        #     canvas = canvas.permute(2, 1, 0).contiguous()
        #     batched_canvas.append(canvas)
        # batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas

class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs

# class Neck(nn.Module):
#     def __init__(self, in_channels, upsample_strides, out_channels):
#         super().__init__()
#         assert len(in_channels) == len(upsample_strides)
#         assert len(upsample_strides) == len(out_channels)
#
#         self.decoder_blocks = nn.ModuleList()
#         for i in range(len(in_channels)):
#             decoder_block = []
#             decoder_block.append(nn.ConvTranspose2d(in_channels[i],
#                                                     out_channels[i],
#                                                     upsample_strides[i],
#                                                     stride=upsample_strides[i],
#                                                     bias=False))
#             decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
#             decoder_block.append(nn.ReLU(inplace=True))
#
#             self.decoder_blocks.append(nn.Sequential(*decoder_block))
#
#         # in consitent with mmdet3d
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def forward(self, x):
#         '''
#         x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
#         return: (bs, 384, 248, 216)
#         '''
#         outs = []
#         for i in range(len(self.decoder_blocks)):
#             xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
#             outs.append(xi)
#         out = torch.cat(outs, dim=1)
#
#         return out

class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()

        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return:
              bbox_cls_pred: (bs, n_anchors*3, 248, 216)
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
# class SpatialGroupConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, groups=1, num_groups=1):
#         super(SpatialGroupConv2d, self).__init__()
#         self.num_groups = num_groups
#         self.group_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, groups=groups),
#                                                        nn.BatchNorm2d(out_channels),
#                                                        nn.ReLU(inplace=True)),
#                                          nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation_rate, \
#                                                                  padding=(kernel_size + (kernel_size - 1) * (dilation_rate - 1) - 1) // 2, groups=groups),
#                                                        nn.BatchNorm2d(out_channels),
#                                                        nn.ReLU(inplace=True))])
#         self.fusion_conv = nn.Sequential(nn.Conv2d(out_channels*2, out_channels, kernel_size=1, padding=0, groups=groups),
#                                          nn.BatchNorm2d(out_channels),
#                                          nn.ReLU(inplace=True))
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         group_size = [width // self.num_groups[0], width // self.num_groups[1]]
#         group_feature = []
#
#         for i in range(len(group_size)):
#             # Split the input tensor into groups along the height dimension
#             y = x.view(batch_size, channels, height, self.num_groups[i], group_size[i])
#
#             # Apply the convolution to each group
#             y = y.permute(0, 3, 1, 2, 4).contiguous()  # (batch_size, num_groups, channels, height, group_size)
#             y = y.view(batch_size * self.num_groups[i], channels, height, group_size[i])
#             y = self.group_conv[i](y)
#             _, out_channels, out_group_size, out_width = y.size()
#             y = y.view(batch_size, self.num_groups[i], out_channels, out_group_size, out_width)
#             y = y.permute(0, 2, 3, 1, 4).contiguous()
#             group_feature.append(y.view(batch_size, out_channels, height, width))
#
#         x = torch.cat(group_feature, dim=1)
#         # Reshape the output back to the original shape
#         #_, out_channels, out_group_size, out_width = x.size()
#         #x = x.view(batch_size, self.num_groups, out_channels, out_group_size, out_width)
#         #x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch_size, out_channels, num_groups, out_group_size, out_width)
#         #x = self.fusion_conv(x.view(batch_size, out_channels, height, width))
#         x = self.fusion_conv(x)
#         return x

class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3,
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.split_sizes = [64, 32, 128, 204]
        self.b = 1
        self.gamma = 2
        self.nclasses = nclasses
        self.pillar_layer = PillarLayer(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=9,
                                            out_channel=64)

        #self.SpatialGroupConv2d = SpatialGroupConv2d(65, 64, kernel_size=3, dilation_rate=2, groups=1, num_groups=[4, 3])

        # self.backbone = Backbone(in_channel=64,
        #                          out_channels=[64, 128, 256],
        #                          layer_nums=[3, 5, 5])

        self.backbone = build_backbone('resnet18', num_classes=1)
        self.neck = RPN(
            layer_nums=[2, 2, 2],
            ds_layer_strides=[1, 2, 2],
            us_num_filters=[128, 256, 512],
            # #logger=logging.getLogger("RPN"),
            )
        # self.neck = Neck(in_channels=[128, 256, 512],
        #                  upsample_strides=[1, 2, 4],
        #                  out_channels=[128, 128, 128])
        self.head = Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)

        # anchors
        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)
        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216)
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return:
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, )
        '''
        # 0. pre-process
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1)  # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d,
                                 scores=cur_bbox_cls_pred,
                                 thresh=self.nms_thr,
                                 pre_maxsize=None,
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(
                cur_bbox_pred)  # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return:
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ]
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i],
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                                                      anchors=batched_anchors[i])
            results.append(result)
    #     return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_feature = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        #pillar_features = self.SpatialGroupConv2d(pillar_feature)

        #pillar_features = self.dcn(pillar_feature) + pillar_feature

        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_feature)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        # bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
                                               batched_gt_bboxes=batched_gt_bboxes,
                                               batched_gt_labels=batched_gt_labels,
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)

            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred,
                                                bbox_pred=bbox_pred,
                                                bbox_dir_cls_pred=bbox_dir_cls_pred,
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred,
                                                bbox_pred=bbox_pred,
                                                bbox_dir_cls_pred=bbox_dir_cls_pred,
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError