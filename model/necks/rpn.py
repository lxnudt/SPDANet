import numpy as np
import torch
from torch import nn
#from ops import DeformConv2d
from torch.nn import functional as F

class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        us_num_filters,
        norm_cfg=None,
        #logger=None,
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._layer_nums = layer_nums
        self._num_upsample_filters = us_num_filters

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        self._upsample_start_idx = len(self._layer_nums) - len(self._num_upsample_filters)

        # must_equal_list = []
        # for i in range(len(self._upsample_strides)):
        #     # print(upsample_strides[i])
        #     must_equal_list.append(
        #         self._upsample_strides[i]
        #         / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
        #     )
        #
        # for val in must_equal_list:
        #     assert val == must_equal_list[0]

        self.block_2 = self._make_layer(
            self._num_upsample_filters[0] * 2,
            384,
            self._layer_nums[0],
            stride=1,
            dilation=2,
            kernel_size=3,
            name='block_2')
        #self.def_2 = DeformConv2d(self._num_upsample_filters[0], self._num_upsample_filters[0], kernel_size=3, stride=1, padding=1, bias=False, modulation=True)

        self.block_3 = self._make_layer(
            self._num_upsample_filters[1] * 2,
            self._num_upsample_filters[0],
            self._layer_nums[1],
            stride=1,
            dilation=2,
            kernel_size=3,
            name='block_3')
        #self.def_3 = DeformConv2d(self._num_upsample_filters[1], self._num_upsample_filters[1], kernel_size=3, stride=1, padding=1, bias=False, modulation=True)

        self.block_4 = self._make_layer(
            self._num_upsample_filters[2],
            self._num_upsample_filters[1],
            self._layer_nums[2],
            stride=1,
            dilation=2,
            kernel_size=3,
            name='block_4')
        #self.def_fusion = DeformConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False, modulation=True)
        self.def_fusion = nn.Sequential(
            nn.Conv2d(384, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(),
            #DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False, modulation=True),
            nn.ConvTranspose2d(32, 384, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.ReLU())
        #self.fusion_block = nn.Conv2d(self._num_upsample_filters[0] + self._num_upsample_filters[1] + self._num_upsample_filters[2], 384, kernel_size=3, stride=1, padding=1, bias=False)


    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, dilation=2, kernel_size=3, name='block_2'):
        padding = dilation * (kernel_size-1) // 2
        block = nn.Sequential(
            #nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            #build_norm_layer(self._norm_cfg, planes)[1],
            nn.ReLU())
        for j in range(num_blocks):
            block.add_module(f"{name}_conv_{j}", nn.Conv2d(planes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False))
            block.add_module(f"{name}_relu_{j}_", nn.ReLU())
        if name != "block_2":
            block.add_module(f"{name}_deconv_{j+1}", nn.ConvTranspose2d(planes, planes, 2, stride=2, bias=False))
        return block

    def forward(self, pillar_features):
        up_block_4 = self.block_4(pillar_features[2])
        up_block_3 = self.block_3(torch.cat([up_block_4, pillar_features[1]], dim=1))
        up_block_2 = self.block_2(torch.cat([up_block_3, pillar_features[0]], dim=1))
        x = self.def_fusion(up_block_2) + up_block_2
        return x
