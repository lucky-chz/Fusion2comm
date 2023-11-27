# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn_occ import Where2comm_occ
import torch
import os
import cv2
import seaborn as sns
import mmcv
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from torchvision import transforms
from opencood.utils.pcd_utils import inImage
from deformable_attention import DeformableAttention

class PointPillarWhere2commLogitsFuse3DDAAddOcclusion(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commLogitsFuse3DDAAddOcclusion, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.seg_loss = getattr(args, 'seg_loss', 'KL_Loss') 
        self.seg_loss_weights = getattr(args, 'seg_loss_weights', 1.0) 
        
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
        self.fusion_net = Where2comm_occ(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.occlusion_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.logits_head = nn.Sequential(
            nn.Linear(self.pillar_vfe.get_output_feature_dim(), self.pillar_vfe.get_output_feature_dim()//2),
            nn.BatchNorm1d(self.pillar_vfe.get_output_feature_dim()//2),
            nn.GELU(),
            nn.Linear(self.pillar_vfe.get_output_feature_dim()//2, 19),
        )
        self.pillar_feat_down = nn.Sequential(
            nn.Linear(self.pillar_vfe.get_output_feature_dim()+19, self.pillar_vfe.get_output_feature_dim()),
            nn.BatchNorm1d(self.pillar_vfe.get_output_feature_dim()),
            nn.GELU(),
            nn.Linear(self.pillar_vfe.get_output_feature_dim(), self.pillar_vfe.get_output_feature_dim()),
        )
        self.logits_atten = nn.Sequential(
                nn.Linear(self.pillar_vfe.get_output_feature_dim()+19+19, self.pillar_vfe.get_output_feature_dim()//2),
                nn.BatchNorm1d(self.pillar_vfe.get_output_feature_dim()//2),
                nn.GELU(),
                nn.Linear(self.pillar_vfe.get_output_feature_dim()//2, 2),
                nn.Softmax(dim=-1)
            )
        self.logits_img_da = nn.Sequential(
            nn.Conv2d(19, self.pillar_vfe.get_output_feature_dim(), kernel_size=1),
            DeformableAttention(
                dim = self.pillar_vfe.get_output_feature_dim(),                    # feature dimensions
                dim_head = self.pillar_vfe.get_output_feature_dim()//8,               # dimension per head
                heads = 2,                   # attention heads
                dropout = 0.,                # dropout
                downsample_factor = 16,       # downsample factor (r in paper)
                offset_scale = None,            # scale of offset, maximum offset
                offset_groups = None,        # number of offset groups, should be multiple of heads
                offset_kernel_size = 16,      # offset kernel size
            ),
            nn.Conv2d(self.pillar_vfe.get_output_feature_dim(), self.pillar_vfe.get_output_feature_dim()//2, kernel_size=1),
            nn.BatchNorm2d(self.pillar_vfe.get_output_feature_dim()//2),
            nn.GELU(),
            nn.Conv2d(self.pillar_vfe.get_output_feature_dim()//2, 19, kernel_size=1),
        )
        if args['backbone_fix']:
            self.backbone_fix()

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
        for p in self.occlusion_head.parameters():
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

    def forward(self, data_dict, vis=False, fuse_2d=True):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        lidar_np = data_dict['lidar_np']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        image_input = data_dict['image_inputs']
        occlusion_v = data_dict['occlusion_state_v']
        occlusion_i = data_dict['occlusion_state_i']
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'lidar_np': lidar_np,
                      'occlusion_v': occlusion_v,
                      'occlusion_i': occlusion_i,
                      'image_inputs': image_input,}
        batch_size = voxel_coords[:, 0].max() + 1
        layer_out = batch_dict['image_inputs']['semseg'] # 可以加一层deformable attention 2d
        # lidar_seg = batch_dict['lidar_np'][:, 4]
        layer_out = torch.cat(layer_out, dim=0)
        layer_out = self.logits_img_da(layer_out) + layer_out
        sem_logits_feature = []
        uv_nonzero_all = []
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
        
        palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        palette = np.array(palette)
        seg_lidar = []
        for batch_idx in range(0, batch_size):
            # layer_out_b = layer_out[batch_idx]
            layer_out_b = layer_out[batch_idx, :, :, :].unsqueeze(0)
            semSegCls = layer_out_b.argmax(axis=1)[0, :, :].cpu().numpy().astype(np.uint8)
            
            pad_zero = torch.ones((1, 1, layer_out_b.shape[2], layer_out_b.shape[3])).to(layer_out_b) * min(-1e10, layer_out_b.min().item())
            layer_out_b = torch.cat([pad_zero, layer_out_b], dim=1)
            
            batch_mask = (voxel_coords[:, 0] == batch_idx)
            points_b = voxel_features[batch_mask]  #  用 batch_mask select voxel_features
            uv_b = points_b[:, :, 5:7]
            seg_lidar_b = points_b[:, :, 4]
            
            seg_lidar.append(seg_lidar_b)
            front_mask = points_b[:, :, 0]>0
            
            mask_in = inImage(uv_b.reshape(-1, 2), 
                              batch_dict['image_inputs']['imgs'][0].shape[2],
                              batch_dict['image_inputs']['imgs'][0].shape[3])
            
            mask_in = mask_in.reshape(points_b.shape[0], points_b.shape[1])
            mask_uv = front_mask & mask_in
            mask_uv_num = mask_uv.sum(dim=1) # - (points_b.shape[1] - voxel_num_points[batch_mask])
            uv_nonzero = mask_uv_num>0
            
            uv = grid = (points_b[:, :, 5:] * mask_uv.unsqueeze(-1)).sum(dim=1)[uv_nonzero]/mask_uv_num[uv_nonzero].unsqueeze(1)
            points_b_nonzero = (points_b[:, :, 0:3] * mask_uv.unsqueeze(-1)).sum(dim=1)[uv_nonzero]/mask_uv_num[uv_nonzero].unsqueeze(1)
            points_b_mean = points_b[:, :, :3].sum(dim=1)/voxel_num_points[batch_mask].unsqueeze(1) # N, 2
            # uv = grid = points_b[:, :, 4:].sum(dim=1)/voxel_num_points[batch_mask].unsqueeze(1) # N, 2
            # uv = uv.long()
            # points = points_b.sum(dim=1)[:,:3]/voxel_num_points[batch_mask].unsqueeze(1)
            grid = grid.unsqueeze(0).unsqueeze(0) # 1,1,N,2
            # grid = torch.flip(grid, dims=[-1])
            grid[:, :, :, 0] = (grid[:, :, :, 0]/batch_dict['image_inputs']['imgs'][0].shape[3]) * 2 - 1 #(1, 1, N, 2)   batch_dict['image_shape']['H']原始图像的W
            grid[:, :, :, 1] = (grid[:, :, :, 1]/batch_dict['image_inputs']['imgs'][0].shape[2]) * 2 - 1 #(1, 1, N, 2)    batch_dict['image_shape']['H']原始图像的H
            sample_data = torch.nn.functional.grid_sample(layer_out_b, grid=grid).squeeze(0).squeeze(1).transpose(0, 1) # 1, logits, 1, N -> logits, 1, N -> logits, N -> N, logits
            sem_logits_b = torch.zeros((points_b.shape[0], layer_out_b.shape[1])).to(sample_data)
            sem_logits_b[uv_nonzero] = sample_data
            sem_logits_feature.append(sem_logits_b)
            uv_nonzero_all.append(uv_nonzero)
            if vis is True:
                output_dir = "/space/chuhz/workspace/v2x_object/visualization"
                ply_file_path = os.path.join(output_dir, f"point_ori_cityscapes_{batch_idx}.ply")                 
                img_file = os.path.join(output_dir, f"image_seg_cityscapes_new_{batch_idx}.jpg")
                                
                image = batch_dict['image_inputs']['imgs'][batch_idx].cpu().numpy()
                infra_img = image.reshape(image.shape[2], image.shape[3], image.shape[1])
                infra_img = np.asarray(infra_img)
                semSegCls = cv2.resize(semSegCls,(batch_dict['image_inputs']['imgs'][0].shape[3], batch_dict['image_inputs']['imgs'][0].shape[2]))
                semSeg_color = palette[semSegCls]
                cv2.imwrite(img_file, (semSeg_color * 0.5 + 0.5 * infra_img).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                resize = transforms.Resize((image.shape[2], image.shape[3]))
                resized_image = resize(layer_out_b).argmax(dim=1)  #.cpu().detach().numpy()
                # color = sample_data.argmax(dim=1).view(-1, 1)
                point_cloud = o3d.geometry.PointCloud()
                # point_cloud.points = o3d.utility.Vector3dVector(points.cpu().numpy())            
                # color_values = color.cpu().numpy()
                # colors = np.array([color_map[color_value[0]] for color_value in color_values])
                # colors = color_map_list[resized_image[:, uv[:, 1], uv[:, 0]].reshape(-1).cpu().numpy()]
                
                point_cloud.points = o3d.utility.Vector3dVector(points_b_mean.cpu().numpy())
                # color_values = resized_image[:, uv_ori[:, 1], uv_ori[:, 0]].reshape(-1).cpu().numpy()
                sem_cls_b = -torch.ones(points_b.shape[0]).to(sample_data).long()
                sem_cls_b[uv_nonzero] = sample_data.argmax(dim=1) - 1
                color_values = sem_cls_b.detach().cpu().numpy()
                colors = np.array([palette[color_value] for color_value in color_values])/255.0
                # colors = color_map_list[resized_image[:, uv_ori[:, 1], uv_ori[:, 0]].reshape(-1).cpu().numpy()]
                point_cloud.colors = o3d.utility.Vector3dVector(colors)                
                o3d.io.write_point_cloud(ply_file_path, point_cloud)
                
                
                self.show_result(image.detach().cpu().transpose(1,2).transpose(2,3).numpy().astype(np.uint8), 
                                resized_image.cpu().detach().numpy().astype(np.uint8), 
                                out_file=img_file, 
                                palette=palette)
        # output_dir = "/space/chuhz/workspace/v2x_object/visualization"
        # os.makedirs(output_dir, exist_ok=True)
        # for idx, sample in enumerate(sem_logits_feature):
        #     plt.figure()
        #     plt.imshow(sample.cpu().detach().numpy(), cmap='viridis')
        #     plt.colorbar()
        #     plt.title(f"Sample Data Visualization for Batch {idx}")
        #     # 保存图像到文件
        #     output_file = os.path.join(output_dir, f"batch_{idx}_visualization.png")
        #     plt.savefig(output_file)
            
        #     # 关闭当前图像，以避免内存泄漏
        #     plt.close()
        
        sem_logits_feature = torch.cat(sem_logits_feature, dim=0)[:, 1:]
        seg_lidar = torch.cat(seg_lidar, dim=0).long()
        uv_nonzero_all = torch.cat(uv_nonzero_all, dim=0)
        # seg_mask_3d = seg_lidar > 1
        seg_lidar_one_hot = torch.nn.functional.one_hot(seg_lidar, 21) 
        one_hot_sum = seg_lidar_one_hot.sum(dim=1)[:,1:]
        one_hot_sum_max = one_hot_sum.argmax(dim=1, keepdim=True)-1
        seg_mask_3d = (one_hot_sum_max >= 0).view(-1)
        
        seg_lidar = seg_lidar
        # voxel_features = torch.cat([voxel_features, sem_logits_feature], dim=-1) # 4 + 2 + C, xyz intensity uv logits
        voxel_features = voxel_features[:, :, :4]
        batch_dict['voxel_features'] = voxel_features
        # 对N，6的feature进行grid_sample
        # 
        # 
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        points_logits = self.logits_head(batch_dict['pillar_features'])
        
        if self.seg_loss == 'KL_Loss': 
            kl_loss = F.kl_div(
                F.log_softmax(points_logits[uv_nonzero_all], dim=1),
                F.softmax(sem_logits_feature[uv_nonzero_all], dim=1),
            )*self.seg_loss_weights
            
            seg_loss = F.cross_entropy(
                sem_logits_feature[seg_mask_3d],
                one_hot_sum_max[seg_mask_3d].reshape(-1),
            )
            kl_loss = kl_loss + seg_loss
        elif self.seg_loss == 'cross_entropy':
            kl_loss = F.cross_entropy(
                points_logits[uv_nonzero_all], 
                F.softmax(sem_logits_feature[uv_nonzero_all], dim=1),
            )*self.seg_loss_weights
            
            seg_loss = F.cross_entropy(
                sem_logits_feature[seg_mask_3d],
                one_hot_sum_max[seg_mask_3d].reshape(-1),
            )
            kl_loss = kl_loss + seg_loss
        else:
            assert False, 'type not in segloss list!'
        if fuse_2d:
            logits_atten_input = torch.cat([batch_dict['pillar_features'][uv_nonzero_all],
                                            points_logits[uv_nonzero_all],
                                            sem_logits_feature[uv_nonzero_all]], dim=-1)
            l_atten = self.logits_atten(logits_atten_input)
            points_logits[uv_nonzero_all] = points_logits[uv_nonzero_all] * l_atten[:, 0].unsqueeze(1)+ l_atten[:, 1].unsqueeze(1) * sem_logits_feature[uv_nonzero_all]
        pillar_features = torch.cat([batch_dict['pillar_features'], 
                                     points_logits], dim=-1)
        batch_dict['pillar_features'] = self.pillar_feat_down(pillar_features)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
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
        # 车端、路端单独的
        psm_single = self.cls_head(spatial_features_2d)
        occ_single = self.occlusion_head(spatial_features_2d)
        op_single = (psm_single[:][1:2]*torch.softmax(occ_single[:][1:2], dim=-1)).view(2, 1, psm_single.shape[2], -1)
        op_single = torch.concat([psm_single[:][0:1].view(2, 1, psm_single.shape[2], -1), op_single], dim=1)
        # op_single = psm_single*torch.softmax(occ_single, dim=-1)
        rm_single = self.reg_head(spatial_features_2d)

        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                            psm_single,
                                            occ_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            op_single,
                                            record_len,
                                            pairwise_t_matrix)
            
            
        print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        occm = self.occlusion_head(fused_feature)
        # print('occm: ', occm.shape)
        # # opencv的归一化、热力图。 cls的真值也可视化一下，做一个对比； 融合后的
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(occm.cpu().numpy(), cmap='YlGnBu', cbar=True, annot=True, fmt=".2f", linewidths=.5)
        # plt.title('Class Confidence Map')
        # plt.xlabel('Class Index')
        # plt.ylabel('Sample Index')
        # plt.tight_layout()

        
        
        # 选择 tensor 的第一个和第二个维度的特定索引
        selected_data = occm[0, 0].cpu().numpy()  # 这将给我们一个形状为 [100, 252] 的数组

        # 使用 seaborn 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(selected_data, cmap='viridis')
        plt.title('Heatmap of Selected Tensor Slice')
        plt.xlabel('Dimension 3')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        # 保存 heatmap 到指定路径
        save_path = '/space/chuhz/workspace/results/heatmap/ours/' + batch_idx +'.png'
        plt.savefig(save_path, dpi=300)  # dpi 参数可以调整保存的图像的分辨率
        
        # occm.cpu().numpy().save('/space/chuhz/workspace/results/heatmap/ref/occ.npy')
        rm = self.reg_head(fused_feature)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'occ': occm,
                       'logits_kl_loss': kl_loss,
                        # 'seg_loss': seg_loss
                       }
        output_dict.update(result_dict)
        
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        split_occ_single = self.regroup(occ_single, record_len)
        psm_single_v = []
        psm_single_i = []
        occ_single_v = []
        occ_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            occ_single_v.append(split_occ_single[b][0:1])
            occ_single_i.append(split_occ_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        occ_single_v = torch.cat(occ_single_v, dim=0)
        occ_single_i = torch.cat(occ_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update({'psm_single_v': psm_single_v,
                       'psm_single_i': psm_single_i,
                       'occ_single_v': occ_single_v,
                       'occ_single_i': occ_single_i,
                       'rm_single_v': rm_single_v,
                       'rm_single_i': rm_single_i,
                       'comm_rate': communication_rates
                       })
        return output_dict
