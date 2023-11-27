# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter, PointPillarScatterv2
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.spvcnn.spvcnn import get_model as SPVCNN
from opencood.models.instance_center_modules.instance_center import InstanceCenter, InstanceComm, InstanceCommClusterPointsFull
from opencood.models.sub_modules.fbg_generation import ForegroundMaskGenerator
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
import torch
import os
import cv2
import mmcv
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from torchvision import transforms
from opencood.utils.pcd_utils import inImage
from copy import deepcopy
from spconv.pytorch.utils import PointToVoxel
class SPVCNNResInstance2commClusterPoints3Full(nn.Module):
    def __init__(self, args, train=True):
        super(SPVCNNResInstance2commClusterPoints3Full, self).__init__()
        self.args = deepcopy(args)
        self.spvcnn_config = deepcopy(args['spvcnn'])
        self.spvcnn = SPVCNN(args['spvcnn'])
        self.instance_center = InstanceCenter(args['InstanceCenter'])
        self.fg_mask_generator = ForegroundMaskGenerator(args['ForegroundMaskGenerator'])
        self.instance2comm = InstanceCommClusterPointsFull(args['InstanceComm'])
        self.voxel_generator = PointToVoxel(
            vsize_xyz=args['voxel_size'],
            coors_range_xyz=args['lidar_range'],
            max_num_points_per_voxel=args['max_points_per_voxel'],
            max_num_voxels=args['max_voxel_train'] if train else args['max_voxel_test'],
            num_point_features=args['num_point_features'], # 这里的feature数目要改掉 改成6
        )
        # PIllar VFE
        self.pillar_vfe_spvcnn = PillarVFE(args['pillar_vfe'],
                                    num_point_features=args['num_point_features'],
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatterv2(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.gate_fusion_pillar = nn.Sequential(
            nn.Conv2d(args['point_pillar_scatter']['num_features']*2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(args['fusion_args']['agg_operator']['feature_dim']*2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()
    def train(self, mode: bool = True):
        args = self.args
        device = list(self.pillar_vfe.parameters())[0].device
        self.voxel_generator = PointToVoxel(
            vsize_xyz=args['voxel_size'],
            coors_range_xyz=args['lidar_range'],
            max_num_points_per_voxel=args['max_points_per_voxel'],
            max_num_voxels=args['max_voxel_train'] if mode else args['max_voxel_test'],
            num_point_features=args['num_point_features'], # 这里的feature数目要改掉 改成6
            device=device
        )
        return super().train(mode)
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def show_result(
                self,
                img,
                result,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        # img = mmcv.imread(img)
        # img = img.copy()
        img = img[0]
        seg = result[0]
        if palette is None:
            
            # Get random state before set seed,
            # and restore random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(
                0, 255, size=(result.shape[1], 3))
            np.random.set_state(state)

        palette = np.array(palette)
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
    
            return img
    def generate_spvcnn_dict(self, data_dict, side):
        if side == 'i':
            pc_proj = data_dict['origin_projected_lidar_i']
            pc = data_dict['origin_unprojected_lidar_i']
            bc_idx = data_dict['origin_lidar_batchidx_i']
            instance_labels = data_dict['instance_labels_i']
            instance_center = data_dict['instance_center_i']
        elif side == 'v':
            pc_proj = data_dict['origin_projected_lidar_v']
            pc = data_dict['origin_unprojected_lidar_v']
            bc_idx = data_dict['origin_lidar_batchidx_v']
            instance_labels = data_dict['instance_labels_v']
            instance_center = data_dict['instance_center_v']
        labels = pc[:, -1].long() - 2
        labels[labels<0] = 1
        pc = pc[:, :-1]
        min_volume_space = self.spvcnn_config['dataset_params']['min_volume_space']
        max_volume_space = self.spvcnn_config['dataset_params']['max_volume_space']
        coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                           [min_volume_space[1], max_volume_space[1]],
                           [min_volume_space[2], max_volume_space[2]]]
        mask =   (pc[:, 0]>coors_range_xyz[0][0]) & (pc[:, 0]<coors_range_xyz[0][1])\
               & (pc[:, 1]>coors_range_xyz[1][0]) & (pc[:, 1]<coors_range_xyz[1][1])\
               & (pc[:, 2]>coors_range_xyz[2][0]) & (pc[:, 2]<coors_range_xyz[2][1])
        data_dict = {
            'points_proj': pc_proj[mask].float(),
            'points': pc[mask].float(),
            'batch_idx': bc_idx[mask].long(),
            'labels': labels[mask].long(),
            'instance_label': instance_labels[mask].long(),
            'center_label': instance_center[mask].float(),
            'batch_size': bc_idx[mask].max().item() + 1,
            'side': side
        }
        return data_dict
    def merge_spvcnn_dict(self, dict_list):
        merge_dict = {'side': '',
                      'batch_size': 0}
        for dict_side in dict_list:
            for key in dict_side:
                value = dict_side[key]
                if type(value) in [int, str]:
                    merge_dict[key] += dict_side[key]
                    continue
                if key not in merge_dict:
                    merge_dict[key] = []
                if key == 'batch_idx':
                    if dict_side['side'] == 'i':
                        value = value * 2 + 1
                    elif dict_side['side'] == 'v':
                        value = value * 2
                merge_dict[key].append(value)
        for key in merge_dict:
            if type(merge_dict[key]) in [int, str]:
                continue
            merge_dict[key] = torch.cat(merge_dict[key], dim=0)
        return merge_dict
    def sp_voxel_gen(self, data_dict, side='all', filter_pc=False):
        batch_size = data_dict['batch_size']
        voxel_all = []
        voxel_coords_all = []
        voxel_num_points_all = []
        for b_idx in range(0, batch_size):
            mask_b = (data_dict['batch_idx'] == b_idx)
            if filter_pc:
                min_volume_space = self.spvcnn_config['dataset_params']['min_volume_space']
                max_volume_space = self.spvcnn_config['dataset_params']['max_volume_space']
                coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                   [min_volume_space[1], max_volume_space[1]],
                                   [min_volume_space[2], max_volume_space[2]]]
                points_proj = data_dict['points_proj']
                mask_proj =   (points_proj[:, 0]>coors_range_xyz[0][0]) & (points_proj[:, 0]<coors_range_xyz[0][1])\
                       & (points_proj[:, 1]>coors_range_xyz[1][0]) & (points_proj[:, 1]<coors_range_xyz[1][1])\
                       & (points_proj[:, 2]>coors_range_xyz[2][0]) & (points_proj[:, 2]<coors_range_xyz[2][1])
                mask_b = mask_proj & mask_b
            pc = data_dict['points'][mask_b]
            middle_feature = data_dict['points_feature'][mask_b]
            logits = data_dict['logits'][mask_b]
            pcd_feat = torch.cat([pc, middle_feature, logits], dim=1)
            if pcd_feat.shape[0]<=0:
                continue
            voxel_output = self.voxel_generator(pcd_feat)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
            coord_bidx = torch.ones((coordinates.shape[0], 1)).to(coordinates) * b_idx
            coordinates = torch.cat([coord_bidx, coordinates], dim=1)
            voxel_all.append(voxels)
            voxel_coords_all.append(coordinates)
            voxel_num_points_all.append(num_points)
        data_dict['empty'] = False
        if len(voxel_all)<=0:
            data_dict['empty'] = True
            return data_dict
        voxel_all = torch.cat(voxel_all, dim=0)
        voxel_coords_all = torch.cat(voxel_coords_all, dim=0)
        voxel_num_points_all = torch.cat(voxel_num_points_all, dim=0)
        data_dict['voxel_features'] = voxel_all
        assert side in ['v', 'i', 'all'], 'parameter side has wrong value'
        if side == 'v':
            data_dict['voxel_coords'] = voxel_coords_all * 2
        elif side == 'i':
            data_dict['voxel_coords'] = voxel_coords_all * 2 + 1
        else:
            data_dict['voxel_coords'] = voxel_coords_all
        data_dict['voxel_num_points'] = voxel_num_points_all
        return data_dict
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    def get_pair_matrix_t(self, pairwise_t_matrix, H, W):
        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (1 * self.fusion_net.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (1 * self.fusion_net.discrete_ratio * H) * 2
        return pairwise_t_matrix
    def forward(self, data_dict, vis=False):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        # lidar_np = data_dict['lidar_np']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        # image_input = data_dict['image_inputs']
        batch_dict = {'voxel_features': voxel_features[:, :, :4],
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'batch_size': voxel_coords[:, 0].max().item() + 1
                    #   'lidar_np': lidar_np,
                    #   'image_inputs': image_input,
                      }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        spatial_features_res = batch_dict['spatial_features']
        data_dict_v = self.generate_spvcnn_dict(data_dict, side='v')
        data_dict_i = self.generate_spvcnn_dict(data_dict, side='i')
        data_dict_vi = self.merge_spvcnn_dict([data_dict_v, data_dict_i])
        data_dict_vi['points_cloud_timestamp'] = data_dict['pointcloud_timestamp']
        data_dict_vi = self.spvcnn(data_dict_vi)
        # data_dict_v = self.spvcnn(data_dict_v)
        # data_dict_i = self.spvcnn(data_dict_i)
        data_dict_vi = self.sp_voxel_gen(data_dict_vi, side='all')
        batch_dict = {
            'voxel_features': data_dict_vi['voxel_features'],
            'voxel_coords': data_dict_vi['voxel_coords'],
            'voxel_num_points': data_dict_vi['voxel_num_points'],
            'record_len': record_len,
            'batch_size': data_dict_vi['batch_size']
        }
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe_spvcnn(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        spatial_features_cat = torch.cat([batch_dict['spatial_features'], spatial_features_res], dim=1)
        spatial_features_gate_weight = self.gate_fusion_pillar(spatial_features_cat)
        spatial_features_fusion = spatial_features_gate_weight[:, 0, :, :].unsqueeze(1) * spatial_features_res + \
                                  spatial_features_gate_weight[:, 1, :, :].unsqueeze(1) * batch_dict['spatial_features']
        batch_dict['spatial_features'] = spatial_features_fusion
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        # psm_single -> psm_single_ego
        # rm_single -> rm_single_ego
        batch_node_psmsingle = self.regroup(psm_single, record_len)
        batch_node_rmsingle = self.regroup(rm_single, record_len)
        batch_node_sf = self.regroup(spatial_features_2d, record_len)
        psm_single_ego = []
        rm_single_ego = []
        sf_single_ego_v = []
        sf_single_ego_i = []
        B, L = pairwise_t_matrix.shape[:2]
        _, H, W = psm_single.shape[1:]
        pair_mt = self.get_pair_matrix_t(pairwise_t_matrix, H, W)
        for b in range(B):
            N = record_len[b]
            t_matrix = pair_mt[b][:N, :N, :, :]
            psm_single_ego_b = warp_affine_simple(batch_node_psmsingle[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            rm_single_ego_b = warp_affine_simple(batch_node_rmsingle[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            sf_single_ego_b = warp_affine_simple(batch_node_sf[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            psm_single_ego_b = torch.mean(psm_single_ego_b, dim=0)
            rm_single_ego_b = torch.mean(rm_single_ego_b, dim=0)
            sf_single_ego_b_v = sf_single_ego_b[0, : ,: ,:]
            sf_single_ego_b_i = sf_single_ego_b[1, :, :, :]
            psm_single_ego.append(psm_single_ego_b)
            rm_single_ego.append(rm_single_ego_b)
            sf_single_ego_v.append(sf_single_ego_b_v)
            sf_single_ego_i.append(sf_single_ego_b_i)
        psm_single_ego = torch.stack(psm_single_ego)
        rm_single_ego = torch.stack(rm_single_ego)
        sf_single_ego_v = torch.stack(sf_single_ego_v)
        sf_single_ego_i = torch.stack(sf_single_ego_i)
        # instance_comm
        # 这几个也可以尝试潜入到spvcnn的每一个stage里面
        data_dict_vi = self.instance_center(data_dict_vi)
        data_dict_vi = self.fg_mask_generator(data_dict_vi)
        data_dict_vi_cluster = self.instance2comm(data_dict_vi)
        data_dict_vi_cluster = self.sp_voxel_gen(data_dict_vi_cluster, side='all', filter_pc=False)
        
        if data_dict_vi_cluster['empty'] is False:
            batch_dict = {
                'voxel_features': data_dict_vi_cluster['voxel_features'],
                'voxel_coords': data_dict_vi_cluster['voxel_coords'],
                'voxel_num_points': data_dict_vi_cluster['voxel_num_points'],
                'record_len': record_len,
                'batch_size': data_dict_vi['batch_size']
            }
            # n, 4 -> n, c
            batch_dict = self.pillar_vfe_spvcnn(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
        else:
            batch_dict = {}
            batch_spatial_features = []
            batch_size = data_dict_vi['batch_size']
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(self.scatter.num_bev_features,
                                              self.scatter.nz * self.scatter.nx * self.scatter.ny).to(voxel_features)
                batch_spatial_features.append(spatial_feature) 
            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.scatter.num_bev_features *
                                        self.scatter.nz, self.scatter.ny, self.scatter.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
            batch_dict['spatial_features'] = batch_spatial_features
        # # n, 4 -> n, c
        # batch_dict = self.pillar_vfe(batch_dict)
        # # n, c -> N, C, H, W
        # batch_dict = self.scatter(batch_dict)
        if self.multi_scale is False:
            batch_dict = self.backbone(batch_dict)
            # N, C, H', W'. [N, 384, 100, 352]
            spatial_features_2d = batch_dict['spatial_features_2d']
            
            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)
            # dcn
            if self.dcn:
                spatial_features_2d = self.dcn_net(spatial_features_2d)
        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix)
            
            
        # print('fused_feature: ', fused_feature.shape)
        gated_feature = torch.cat([fused_feature, sf_single_ego_v], dim=1)
        gated_w = self.gate_fusion(gated_feature)
        fused_feature_v = (gated_w[:, 0, :, :].unsqueeze(1) * fused_feature + 
                         gated_w[:, 1, :, :].unsqueeze(1) * sf_single_ego_v)
        psm_ego_v = self.cls_head(fused_feature_v)# * 0.5 + psm_single_ego * 0.5
        rm_ego_v = self.reg_head(fused_feature_v)# * 0.5 + rm_single_ego * 0.5
        
        
        gated_feature = torch.cat([fused_feature, sf_single_ego_i], dim=1)
        gated_w = self.gate_fusion(gated_feature)
        fused_feature_i = (gated_w[:, 0, :, :].unsqueeze(1) * fused_feature + 
                         gated_w[:, 1, :, :].unsqueeze(1) * sf_single_ego_i)
        psm_ego_i = self.cls_head(fused_feature_i)# * 0.5 + psm_single_ego * 0.5
        rm_ego_i = self.reg_head(fused_feature_i)# * 0.5 + rm_single_ego * 0.5
        # psm = self.cls_head(fused_feature) * 0.5 + psm_single_ego * 0.5
        # rm = self.reg_head(fused_feature) * 0.5 + rm_single_ego * 0.5

        # output_dict = {'psm': psm_ego_v,
        #                'rm': rm_ego_v,
        #                'psm_ego_i': psm_ego_i,
        #                'rm_ego_i': rm_ego_i,
        #                'points_seg_loss': data_dict_vi['loss']
        #                }
        output_dict = {'psm': psm_ego_v,
                       'rm': rm_ego_v,
                       'psm_ego_i': psm_ego_i,
                       'rm_ego_i': rm_ego_i,
                       'points_seg_loss': data_dict_vi['loss']
                       }
        output_dict.update(result_dict)
        
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': communication_rates
                       })
        return output_dict
