import torch
import torch.nn as nn
import torch.nn.functional as F
class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose1d if transpose else nn.Conv1d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        #self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        #x = self.dropout(self.conv(input))
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=1)
        )

        self.score_fn_point = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=2)
        )
        self.mlp = SharedMLP(in_channels * 2, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N)
        """
        # computing attention scores
        scores_channel = self.score_fn_channel(x.permute(0, 2, 1)).permute(0, 2, 1)
        scores_point = self.score_fn_point(x.permute(0, 2, 1)).permute(0, 2, 1)

        # sum over the neighbors
        #features_channel = torch.sum(scores_channel * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)
        #features_point = torch.sum(scores_point * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)
        #features = torch.cat([features_channel, features_point], dim=1)
        features = torch.cat([scores_channel * x, scores_point * x], dim=1)
        features = self.mlp(features)
        #features = torch.max(features, dim=-1)[0]
        return features
        #return features.unsqueeze(-1) #self.mlp(features).squeeze(0)


# class LocalSpatialEncoding(nn.Module):
#     def __init__(self, channels, alpha, k):
#         super(LocalSpatialEncoding, self).__init__()
#
#         self.channels = channels
#         self.mlp = SharedMLP(channels*2, channels, bn=True, activation_fn=nn.ReLU())
#         self.alpha = alpha
#         self.k = k
#
#     def forward(self, coords, features):
#         r"""
#             Forward pass
#
#             Parameters
#             ----------
#             coords: torch.Tensor, shape (B, N, 3)
#                 coordinates of the point cloud
#             features: torch.Tensor, shape (B, d, N, 1)
#                 features of the point cloud
#             neighbors: tuple
#
#             Returns
#             -------
#             torch.Tensor, shape (B, 2*d, N, K)
#         """
#         # 计算特征余弦相似度矩阵
#         feature_sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
#
#         # 计算空间欧式距离相似度矩阵
#         coords = coords.float()
#         coords_dist = torch.cdist(coords, coords, p=2)
#         coords_sim = 1 / (1 + coords_dist)
#         # idx(B, N, K), coords(B, N, 3)
#
#         combined_scores = self.alpha * feature_sim + (1 - self.alpha) * coords_sim
#         combined_scores.fill_diagonal_(-float('inf'))
#
#         topk_scores, topk_indices = torch.topk(combined_scores, k=self.k, dim=-1)
#
#         sel_features = features[topk_indices]
#         topk_scores = F.softmax(topk_scores, dim=1).unsqueeze(-1).expand(-1, -1, self.channels)
#         sel_features = torch.sum(sel_features * topk_scores, dim=1)
#         concat = torch.cat((sel_features, features), dim=1)
#         return self.mlp(concat.unsqueeze(-1))

class LocalSpatialEncoding(nn.Module):
    def __init__(self, d_out=64, d_in=64, in_ch=1, out_ch=8, base_kernel=3, max_dilation=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch * in_ch * base_kernel ** 2 + 2)  # 权重+水平/垂直空洞率
        )
        self.base_kernel = base_kernel
        self.max_dilation = max_dilation
        self.out_ch = out_ch
        self.in_ch = in_ch

        self.conv2d = nn.Sequential(nn.Conv2d(in_channels=out_ch, out_channels=in_ch, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=False))
        self.mlp = SharedMLP(d_out, d_in, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(0)
        batch = x.size(0)
        params = self.param_net(x)

        # 分解动态参数:ml-citation{ref="6" data="citationList"}
        weights = params[:, :-2].view(batch, self.out_ch, self.in_ch, self.base_kernel, self.base_kernel)
        dilation_h = torch.sigmoid(params[:, -2]) * self.max_dilation
        dilation_v = torch.sigmoid(params[:, -1]) * self.max_dilation

        # 动态卷积计算:ml-citation{ref="3,8" data="citationList"}
        outputs = []
        for b in range(batch):
            conv = nn.Conv2d(self.in_ch, self.out_ch, self.base_kernel,
                             dilation=(int(dilation_h[b]), int(dilation_v[b])),
                             padding=self._calc_padding(int(dilation_h[b]), int(dilation_v[b]))).to(x.device)
            conv.weight.data = weights[b]
            outputs.append(conv(x[b:b + 1]))
        outputs = torch.cat(outputs, dim=0)
        outputs = self.conv2d(outputs).squeeze(0)
        #outputs = torch.max(outputs, dim=1)[0]
        outputs = self.mlp(outputs.permute(2, 1, 0))
        return outputs

    def _calc_padding(self, dh, dv):
        ph = (self.base_kernel - 1) * dh // 2
        pv = (self.base_kernel - 1) * dv // 2
        return (ph, pv)

class FeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, base_kernel, max_dilation):
        super(FeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(4, d_out, bn=True, activation_fn=nn.ReLU())

        self.mlp2 = SharedMLP(d_out*2, d_out*2, bn=True, activation_fn=nn.ReLU())

        #self.shortcut = SharedMLP(d_in, d_out, bn=True)
        self.pool1 = AttentivePooling(d_out, d_out)
        #self.pool2 = AttentivePooling(d_out, d_out)

        self.lse1 = LocalSpatialEncoding(d_out=d_out*2, d_in=d_in, in_ch=1, out_ch=8, base_kernel=base_kernel, max_dilation=max_dilation)

        #self.lse2 = LocalSpatialEncoding(d_out=d_in, d_in=d_in, in_ch=1, out_ch=8, base_kernel=base_kernel, max_dilation=max_dilation)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        coords = coords.float()
        coords = self.mlp1(coords.unsqueeze(-1))
        x_local = self.pool1(features)
        #x_local = self.pool2(x_local)
        x_local = torch.max(x_local, dim=-1)[0]
        x = torch.cat([x_local.unsqueeze(-1), coords], dim=1)
        x = self.mlp2(x)
        #x = self.pool2(x)
        x = self.lse1(x.permute(2, 0, 1))
        #x = self.lse2(x.permute(2, 0, 1))

        # bs = coords[-1, 0] + 1
        # batched_canvas = []
        # for i in range(bs):
        #     cur_coors_idx = coords[:, 0] == i
        #     cur_coors = coords[cur_coors_idx, :3]
        #     cur_features = x[cur_coors_idx]
        #
        #     cur_features = self.lse1(cur_coors, cur_features)
        #     #cur_features = self.lse2(cur_coors, cur_features)
        #     batched_canvas.append(cur_features)
        # x = torch.cat(batched_canvas, dim=1)
        #x = torch.max(x, dim=-1)[0]+torch.max(features, dim=-1)[0]
        #x = self.lrelu(x).squeeze(-1)+torch.max(features, dim=-1)[0] + x_local

        x = x.squeeze(-1)+torch.max(features, dim=-1)[0]+x_local.squeeze(-1)

        return x