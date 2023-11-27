#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: voxel_fea_generator.py
@time: 2021/8/4 13:36
'''
import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, coors_range_pol, spatial_shape, scale_list, type='cat', shuffle_voxel=False):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz
        self.coors_range_pol = coors_range_pol
        self.type = type
        # TO-DO 
        self.shuffle_voxel = shuffle_voxel
    @staticmethod
    def cart2polar(input_xyz):
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
        return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape, quantize_type='cat'):
        if quantize_type == 'cat':
            idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        elif quantize_type == 'cylinder':
            assert pc.shape[1] == 3, 'pc dimention should be (N,3) not ' + str(pc.shape)
            pc_pol = voxelization.cart2polar(pc)
            coors_range = torch.tensor(coors_range).to(pc)
            min_bound = coors_range[:, 0]
            max_bound = coors_range[:, 1]
            # min_bound = torch.min(pc_pol, dim=0)[0]
            # max_bound = torch.max(pc_pol, dim=0)[0]
            crop_range = max_bound - min_bound
            # for i in range(0, 3):
            #     pc_pol[:, i] = torch.clamp(pc_pol[:, i], min_bound[i], max_bound[i])
            spatial_shape = spatial_shape.reshape((1, 3))
            crop_range = crop_range.reshape((1, 3))
            min_bound = min_bound.reshape((1, 3))
            max_bound = max_bound.reshape((1, 3))
            idx = spatial_shape * (pc_pol - min_bound)/crop_range
        else:
            raise NotImplementedError
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        bs = data_dict['batch_idx'].max().item() + 1
        for idx, scale in enumerate(self.scale_list):
            if self.type == 'cat':
                xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale), self.type)
                yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale), self.type)
                zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale), self.type)
            elif self.type == 'cylinder':
                spatial_shape = torch.ceil(torch.tensor(self.spatial_shape)/scale).to(pc)
                idx = torch.zeros((pc.shape[0], 3), dtype=torch.long, device=pc.device)
                for bs_i in range(0, bs):
                    now_mask = (data_dict['batch_idx'] == bs_i)
                    idx_i = self.sparse_quantize(pc[now_mask, :], self.coors_range_pol, spatial_shape, self.type)
                    idx[now_mask, :] = idx_i
                xidx = idx[:, 0]
                yidx = idx[:, 1]
                zidx = idx[:, 2]
            else:
                raise NotImplementedError
            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx, yidx, zidx], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict



class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear((in_channels + 6)*2, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx] # .float()
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1)
        return pc_feature

    def forward(self, data_dict):
        pt_fea_cat = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea_pol = self.prepare_input(
            torch.cat([voxelization.cart2polar(data_dict['points'][:, :3]), data_dict['points'][:, 3:]], dim=1),
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = torch.cat([pt_fea_cat, pt_fea_pol], dim=1)
        pt_fea = self.PPmodel(pt_fea)

        # features = torch_scatter.scatter_mean(pt_fea.float(), data_dict['scale_1']['coors_inv'], dim=0)
        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict