# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

from opt import MyConfig

class BMN_model(nn.Module):

    def __init__(self, opt):

        super(BMN_model, self).__init__()

        self.tscale = opt.temporal_scale                     # T: 100(D: 100)
        self.prop_boundary_ratio = opt.prop_boundary_ratio   # 0.5
        self.num_sample = opt.num_sample                     # 32
        self.num_sample_perbin = opt.num_sample_perbin       # 3
        self.feat_dim=opt.feat_dim                           # input_channel: 400

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module: (conv1d_1 + conv1d_2)
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module(start & end): 2 * (conv1d_3 + conv1d_4)
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module: BM_layer(x_1d_p) + conv3d + conv2d
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_2d, kernel_size=3, padding=1),    
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_2d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid() 
        )

    
    def forward(self, inputs):

        """
        Schematic diagram is as below:

                            inputs(Sf): [batch_size][C][T], [3][400][100] 
                                |
                                |   Base Module(conv1d_1 & 2)
                                |
                            base feature(Sf`): [batch_size][C][T], [3][256][100] 
                                |
                    -------------------------------------------------
                    |                                               |
                TEM Module(conv1d_3 & 4)                        BM layer
                    |                                               |BM feature map(Mf): [batch_size][C][T], [3][128][100]
        temporal boundary probability sequence          _boundary_matching_layer
                                                                    |(Mf`): [batch_size][C][N][D][T], [3][128][32][100][100]
            start: [batch_size][T], [3][100]                    conv_3d_1
            end: [batch_size][T], [3][100]                          |(Mf`): [batch_size][C][D][T], [3][512][100][100]
                                                                onv_2d_1 & 2 & 3
                                                                    |
                                                            BM confidence map(Mc):[batch_size][C][D][T], [3][2][100][100]
        """

        base_feature = self.x_1d_b(inputs)
        
        # Temporal Evaluation Module.
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        
        # Proposal Evaluation Module.
        bm_feature_map = self.x_1d_p(base_feature)
        bm_feature_map = self._boundary_matching_layer(bm_feature_map)
        bm_feature_map = self.x_3d_p(bm_feature_map).squeeze(2)
        bm_confidence_map = self.x_2d_p(bm_feature_map)

        return bm_confidence_map, start, end


if __name__ == "__main__":

    arg = MyConfig()
    arg.parse()

    model = BMN_model(opt=arg)
    # summary(model, (3, 416, 416))