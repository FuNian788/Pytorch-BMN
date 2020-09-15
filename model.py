# coding: utf-8

import math

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from opt import MyConfig


class BMN_model(nn.Module):

    """
    Model for BMN.

    Return Arguements:
        bm_confidence_map: (tensor[batch_size][C=2][D][T]): BM confidence map which consists of 'regression' and 'binary classiication'.
        start: (tensor[batch_size][T]): start score sequence.
        end: (tensor[batch_size][T]): end score sequence.

    """

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

        self._get_sample_mask()

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
                TEM Module(conv1d_3 & 4)                        BM layer(sample)
                    |                                               |BM feature map(Mf): [batch_size][C][T], [3][128][100]
        temporal boundary probability sequence              _boundary_matching_layer
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

    
    def _get_sample_mask(self):
        """
        Generate all the possible proposals' sampling weight masks for Boundary-Matching layer.
            Can be done apart from feature extraction, once you get 'tscale' and 'sample times(N)', you can make masks.

        Return Arguements:
            sample_mask: (np.ndarray[T][N'], [100][320000]): all sample masks.
        """
        mask = []
        for end_index in range(self.tscale):
            mask_ = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    start_proposal, end_proposal = start_index, end_index + 1
                    length_proposal = float(end_proposal - start_proposal) + 1
                    # Expand the proposal and add context feature from both side.
                    start_proposal = start_proposal - length_proposal * self.prop_boundary_ratio
                    end_proposal = end_proposal + length_proposal * self.prop_boundary_ratio
                    mask_.append(self._get_sample_mask_per_proposal(start_proposal, end_proposal,
                                                                    self.tscale, self.num_sample, self.num_sample_perbin))
                else:
                    mask_.append(np.zeros([self.tscale, self.num_sample]))
                    
            # For each end index, add 'tscale' proposal-masks to 'mask_'.
            # before stack, mask_.shape: [start_index][T][N], [100][100][32]
            # after stack, mask_.shape: [T][N][start_index], [100][32][100]
            mask_ = np.stack(mask_, axis=2)  
            mask.append(mask_)

        # before stack, mask.shape: [end_index][T][N][start_index], [100][100][32][100]
        # after stack, mask.shape: [T][N][start_index][end_index], [100][32][100][100]
        mask = np.stack(mask, axis=3).astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask).view(self.tscale, -1), requires_grad=False)
    
    
    def _get_sample_mask_per_proposal(self, start_proposal, end_proposal, tscale, num_sample, num_sample_perbin):
        """
        Generate a sampling weight mask of certain proposal.

        Arguements:
            start_proposal: (float[1]): start time of certain expanded proposal.
            end_proposal: (float[1]): end time of certain expanded proposal.
            tscale: (int[1]): length of whole feature.
            num_sample: (int[1]): number of sample points, 'N' in paper.
            num_sample_perbin: (int[1]): number of further sample times in each sample point.
                NOTE: real sample times = num_sample * num_sample_perbin

        Return Arguements:
            mask: (Tensor[T][N]): sample mask for one proposal.
        """       
        length_proposal = end_proposal - start_proposal
        
        length_sample_perbin = length_proposal / (num_sample * num_sample_perbin - 1.0)
        samples = [start_proposal + length_sample_perbin * i for i in range(num_sample * num_sample_perbin)]

        mask = []
        for i in range(num_sample):
            samples_perbin = samples[i * num_sample_perbin: (i + 1) * num_sample_perbin]
            mask_perbin = np.zeros([tscale])
            for j in samples_perbin:
                j_fractional, j_integral = math.modf(j)
                j_integral = int(j_integral)
                if 0 <= j_integral < (tscale - 1):
                    mask_perbin[j_integral] += 1 - j_fractional
                    mask_perbin[j_integral + 1] += j_fractional 
            mask_perbin = 1.0 / num_sample_perbin * mask_perbin
            mask.append(mask_perbin)
        
        mask = np.stack(mask, axis=1)
        return mask


    def _boundary_matching_layer(self, bm_feature_map):
        """
        For each proposal, through BM layer, conduct dot product at T demension between 
            sampling mask weight and temporal feature sequence to generate BM feature map, whose core is 'sample'.

        Arguements:
            bm_feature_map: (tensor[batch_size][C][T]): boundary-matching feature map.

        Return Arguements:
            output: (tensor[batch_size][C][N][D][T]): sampled feature.
        """
        feature_size = bm_feature_map.size()    # bm_feature_map.shape: [3][128][100]
        output = torch.matmul(bm_feature_map, self.sample_mask).reshape(feature_size[0], feature_size[1], 
                                                                        self.num_sample, self.tscale, self.tscale)
        return output


if __name__ == "__main__":

    arg = MyConfig()
    arg.parse()

    model = BMN_model(opt=arg)

    # inputs = torch.randn(3, 400, 100)
    # bm_confidence_map, start, end = model(inputs)
    # print(bm_confidence_map.shape)  # torch.Size([3, 2, 100, 100])
    # print(start.shape)              # torch.Size([3, 100])
    # print(end.shape)                # torch.Size([3, 100])

    # summary(model, (400, 100))  