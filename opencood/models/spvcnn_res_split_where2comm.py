# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.spvcnn.spvcnn import get_model as SPVCNN
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

class SPVCNNResSplitWhere2comm(nn.Module):
    def __init__(self, args, train=True):
        super(SPVCNNResSplitWhere2comm, self).__init__()
        self.args = deepcopy(args)
        self.spvcnn_config = deepcopy(args['spvcnn'])
        self.spvcnn_vehicle = SPVCNN(args['spvcnn'])
        self.spvcnn_infra = SPVCNN(args['spvcnn'])
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
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 128)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 128)
        self.gate_fusion = nn.Sequential(
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
            pc = data_dict['origin_unprojected_lidar_i']
            bc_idx = data_dict['origin_lidar_batchidx_i']
        elif side == 'v':
            pc = data_dict['origin_unprojected_lidar_v']
            bc_idx = data_dict['origin_lidar_batchidx_v']
        labels = pc[:, -1].long() - 2
        labels[labels<0] = 0
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
            'points': pc[mask].float(),
            'batch_idx': bc_idx[mask].long(),
            'labels': labels[mask].long(),
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
                if 'loss' in key:
                    if key not in merge_dict:
                        merge_dict[key] = dict_side[key]
                    else:
                        merge_dict[key] += dict_side[key]
                    continue
                if type(value) is not torch.Tensor:
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
            if 'loss' in key:
                continue
            merge_dict[key] = torch.cat(merge_dict[key], dim=0)
        return merge_dict
    def sp_voxel_gen(self, data_dict, side='all'):
        batch_size = data_dict['batch_size']
        voxel_all = []
        voxel_coords_all = []
        voxel_num_points_all = []
        for b_idx in range(0, batch_size):
            mask_b = (data_dict['batch_idx'] == b_idx)
            pc = data_dict['points'][mask_b]
            middle_feature = data_dict['points_feature'][mask_b]
            logits = data_dict['logits'][mask_b]
            pcd_feat = torch.cat([pc, middle_feature, logits], dim=1)
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
                    #   'lidar_np': lidar_np,
                    #   'image_inputs': image_input,
                      }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        spatial_features_res = batch_dict['spatial_features']
        data_dict_v = self.generate_spvcnn_dict(data_dict, side='v')
        data_dict_i = self.generate_spvcnn_dict(data_dict, side='i')
        data_dict_v = self.spvcnn_vehicle(data_dict_v)
        data_dict_i = self.spvcnn_infra(data_dict_i)
        data_dict_vi = self.merge_spvcnn_dict([data_dict_v, data_dict_i])
        # data_dict_v = self.spvcnn(data_dict_v)
        # data_dict_i = self.spvcnn(data_dict_i)
        data_dict_vi = self.sp_voxel_gen(data_dict_vi, side='all')
        batch_dict = {
            'voxel_features': data_dict_vi['voxel_features'],
            'voxel_coords': data_dict_vi['voxel_coords'],
            'voxel_num_points': data_dict_vi['voxel_num_points'],
            'record_len': record_len
        }
        # batch_size = voxel_coords[:, 0].max() + 1
        # layer_out = batch_dict['image_inputs']['semseg'] # 可以加一层deformable attention 2d
        # sem_logits_feature = []
        
        # 根据给定的颜色值为点云中的每个点分配颜色
        # color_map = {i: np.random.rand(3) for i in range(20)}        
        # color_map_list = [np.random.rand(1, 3) for i in range(20)]  #{i: np.random.rand(3) for i in range(20)}        
        # color_map_list = np.concatenate(color_map_list, axis=0)
        # color_map = {
        #     -1: (0,0,0),
        #     0: (255, 0, 255),   # 道路 紫色
        #     1: (128, 138, 135),   # 人行道 冷灰
        #     2: (156, 102, 31),     # 建筑物 砖红
        #     3: (255, 99, 71),  # 墙壁 番茄红
        #     4: (25, 25, 112),  # 栏杆 深蓝
        #     5: (107, 142, 35),  # 障碍物 草绿色
        #     6: (255, 255, 0),   # 交通信号灯 黄色
        #     7: (255, 0, 0),    # 行人 红色
        #     8: (0, 255, 0),   # 行人 绿色
        #     9: (34, 139, 34),  # 植被 森林绿
        #     10: (218, 112, 214),  # 地面 淡紫色
        #     11: (0, 199, 140),   # 道路标志 土耳其蓝
        #     12: (0, 0, 255),     # 车辆 蓝色
        #     13: (64, 224, 205),     # 水域 青绿色
        #     14: (192, 192, 192),      # 天空 灰色
        #     15: (128, 42, 42),    # 桥梁 棕色
        #     16: (0, 80, 100),    # 隧道
        #     17: (0, 0, 230),     # 轨道
        #     18: (0, 255, 255),   # 路边停车位
        #     19: (0, 0, 0)        # 未标记
        # }
        # palette = []
        # for i in color_map.keys():
        #     if i<0:
        #         continue
        #     palette.append(color_map[i])
        
        # palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        #        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        #        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        #        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        #        [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        # palette = np.array(palette)
        # for batch_idx in range(0, batch_size):
        #     layer_out_b = layer_out[batch_idx]
        #     semSegCls = layer_out_b.argmax(axis=1)[0, :, :].cpu().numpy().astype(np.uint8)
            
        #     pad_zero = torch.ones((1, 1, layer_out_b.shape[2], layer_out_b.shape[3])).to(layer_out_b) * min(-1e10, layer_out_b.min().item())
        #     layer_out_b = torch.cat([pad_zero, layer_out_b], dim=1)
            
        #     batch_mask = (voxel_coords[:, 0] == batch_idx)
        #     points_b = voxel_features[batch_mask]  #  用 batch_mask select voxel_features
        #     uv_b = points_b[:, :, 4:6]
        #     front_mask = points_b[:, :, 0]>0
        #     mask_in = inImage(uv_b.reshape(-1, 2), 
        #                       batch_dict['image_inputs']['imgs'][0].shape[2],
        #                       batch_dict['image_inputs']['imgs'][0].shape[3])
        #     mask_in = mask_in.reshape(points_b.shape[0], points_b.shape[1])
        #     mask_uv = front_mask & mask_in
        #     mask_uv_num = mask_uv.sum(dim=1) # - (points_b.shape[1] - voxel_num_points[batch_mask])
        #     uv_nonzero = mask_uv_num>0
            
        #     uv = grid = (points_b[:, :, 4:] * mask_uv.unsqueeze(-1)).sum(dim=1)[uv_nonzero]/mask_uv_num[uv_nonzero].unsqueeze(1)
        #     points_b_nonzero = (points_b[:, :, 0:3] * mask_uv.unsqueeze(-1)).sum(dim=1)[uv_nonzero]/mask_uv_num[uv_nonzero].unsqueeze(1)
        #     points_b_mean = points_b[:, :, :3].sum(dim=1)/voxel_num_points[batch_mask].unsqueeze(1) # N, 2
        #     # uv = grid = points_b[:, :, 4:].sum(dim=1)/voxel_num_points[batch_mask].unsqueeze(1) # N, 2
        #     # uv = uv.long()
        #     # points = points_b.sum(dim=1)[:,:3]/voxel_num_points[batch_mask].unsqueeze(1)
        #     grid = grid.unsqueeze(0).unsqueeze(0) # 1,1,N,2
        #     # grid = torch.flip(grid, dims=[-1])
        #     grid[:, :, :, 0] = (grid[:, :, :, 0]/batch_dict['image_inputs']['imgs'][0].shape[3]) * 2 - 1 #(1, 1, N, 2)   batch_dict['image_shape']['H']原始图像的W
        #     grid[:, :, :, 1] = (grid[:, :, :, 1]/batch_dict['image_inputs']['imgs'][0].shape[2]) * 2 - 1 #(1, 1, N, 2)    batch_dict['image_shape']['H']原始图像的H
        #     sample_data = torch.nn.functional.grid_sample(layer_out_b, grid=grid).squeeze(0).squeeze(1).transpose(0, 1) # 1, logits, 1, N -> logits, 1, N -> logits, N -> N, logits
        #     sem_logits_b = torch.zeros((points_b.shape[0], layer_out_b.shape[1])).to(sample_data)
        #     sem_logits_b[uv_nonzero] = sample_data
        #     sem_logits_feature.append(sem_logits_b)
            
        #     if vis is True:
        #         output_dir = "/space/chuhz/workspace/v2x_object/visualization"
        #         ply_file_path = os.path.join(output_dir, f"point_ori_cityscapes_{batch_idx}.ply")                 
        #         img_file = os.path.join(output_dir, f"image_seg_cityscapes_new_{batch_idx}.jpg")
                                
        #         image = batch_dict['image_inputs']['imgs'][batch_idx].cpu().numpy()
        #         infra_img = image.reshape(image.shape[2], image.shape[3], image.shape[1])
        #         infra_img = np.asarray(infra_img)
        #         semSegCls = cv2.resize(semSegCls,(batch_dict['image_inputs']['imgs'][0].shape[3], batch_dict['image_inputs']['imgs'][0].shape[2]))
        #         semSeg_color = palette[semSegCls]
        #         cv2.imwrite(img_file, (semSeg_color * 0.5 + 0.5 * infra_img).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
        #         resize = transforms.Resize((image.shape[2], image.shape[3]))
        #         resized_image = resize(layer_out_b).argmax(dim=1)  #.cpu().detach().numpy()
        #         # color = sample_data.argmax(dim=1).view(-1, 1)
        #         point_cloud = o3d.geometry.PointCloud()
        #         # point_cloud.points = o3d.utility.Vector3dVector(points.cpu().numpy())            
        #         # color_values = color.cpu().numpy()
        #         # colors = np.array([color_map[color_value[0]] for color_value in color_values])
        #         # colors = color_map_list[resized_image[:, uv[:, 1], uv[:, 0]].reshape(-1).cpu().numpy()]
                
        #         point_cloud.points = o3d.utility.Vector3dVector(points_b_mean.cpu().numpy())
        #         # color_values = resized_image[:, uv_ori[:, 1], uv_ori[:, 0]].reshape(-1).cpu().numpy()
        #         sem_cls_b = -torch.ones(points_b.shape[0]).to(sample_data).long()
        #         sem_cls_b[uv_nonzero] = sample_data.argmax(dim=1) - 1
        #         color_values = sem_cls_b.detach().cpu().numpy()
        #         colors = np.array([palette[color_value] for color_value in color_values])/255.0
        #         # colors = color_map_list[resized_image[:, uv_ori[:, 1], uv_ori[:, 0]].reshape(-1).cpu().numpy()]
        #         point_cloud.colors = o3d.utility.Vector3dVector(colors)                
        #         o3d.io.write_point_cloud(ply_file_path, point_cloud)
                
                
        #         self.show_result(image.detach().cpu().transpose(1,2).transpose(2,3).numpy().astype(np.uint8), 
        #                         resized_image.cpu().detach().numpy().astype(np.uint8), 
        #                         out_file=img_file, 
        #                         palette=palette)
        # # output_dir = "/space/chuhz/workspace/v2x_object/visualization"
        # # os.makedirs(output_dir, exist_ok=True)
        # # for idx, sample in enumerate(sem_logits_feature):
        # #     plt.figure()
        # #     plt.imshow(sample.cpu().detach().numpy(), cmap='viridis')
        # #     plt.colorbar()
        # #     plt.title(f"Sample Data Visualization for Batch {idx}")
        # #     # 保存图像到文件
        # #     output_file = os.path.join(output_dir, f"batch_{idx}_visualization.png")
        # #     plt.savefig(output_file)
            
        # #     # 关闭当前图像，以避免内存泄漏
        # #     plt.close()
        # sem_logits_feature = torch.cat(sem_logits_feature, dim=0).unsqueeze(1).repeat(1, voxel_features.shape[1], 1).argmax(dim=-1, keepdim=True)
        # voxel_features = torch.cat([voxel_features, sem_logits_feature], dim=-1) # 4 + 2 + C, xyz intensity uv logits
        # batch_dict['voxel_features'] = voxel_features
        # # 对N，6的feature进行grid_sample
        # 
        # 
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe_spvcnn(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        spatial_features_cat = torch.cat([batch_dict['spatial_features'], spatial_features_res], dim=1)
        spatial_features_gate_weight = self.gate_fusion(spatial_features_cat)
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
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm,
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
