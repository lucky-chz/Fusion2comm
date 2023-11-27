import torch
import torch.nn as nn
from copy import deepcopy
import torch_scatter

def get_position_embedding(x, embedding_dim, pos_temperature):
    '''
        b
    '''
    pos_length = embedding_dim
    inv_freq = torch.arange(
        pos_length, dtype=torch.float32, device=x.device)
    inv_freq = pos_temperature ** (2 * (inv_freq // 2) / pos_length)

    # [num_tokens, pos_length]
    embed_x = x[:, None] / inv_freq[None, :]

    # [num_tokens, pos_length]
    embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()],
                            dim=-1).flatten(1)
    # [num_tokens, pos_length * 2]
    pos_embed_2d = embed_x
    return pos_embed_2d

class InstanceCenter(nn.Module):
    def __init__(self, args):
        super(InstanceCenter, self).__init__()
        self.args = args
        self.input_dims = self.args['input_dims']
        self.hiddel_dims = self.args['hiddel_dims']
        self.center_reg = nn.Sequential(
            nn.Linear(self.input_dims, self.hiddel_dims),
            nn.BatchNorm1d(self.hiddel_dims),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.hiddel_dims, 3)
        )
        self.reg_loss = torch.nn.SmoothL1Loss()
        self.reg_loss_weight = self.args['reg_loss_weight'] 
    def get_loss(self, points_center, instance_label, center_label):
        reg_mask = instance_label>=0
        if reg_mask.sum()>0:
            center_loss = self.reg_loss(points_center[reg_mask], center_label[reg_mask])
        else:
            center_loss = 0
        return center_loss
    def forward(self, data_dict):
        points_feature = data_dict['points_feature']
        points_center_res = self.center_reg(points_feature)
        center_loss = self.get_loss(points_center_res,
                                    data_dict['instance_label'],
                                    data_dict['center_label'])
        data_dict['center_reg_loss'] = center_loss
        data_dict['points_center_res'] = points_center_res
        data_dict['points_center'] = points_center_res + data_dict['points_proj']
        if 'loss' in data_dict:
            data_dict['loss'] += center_loss*self.reg_loss_weight
        return data_dict

class InstanceComm(nn.Module):
    def __init__(self, args):
        super(InstanceComm, self).__init__()
        self.args = args
        self.voxel_size = self.args['voxel_size']
        self.pc_range = self.args['pc_range']
        self.input_dims = self.args['input_dims']
        self.time_pos_temperature = self.args['time_pos_temperature']
        self.pf_down = nn.Sequential(
            nn.Linear(self.input_dims*3, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
        self.cluster_proj = nn.Sequential(
            nn.Linear(self.input_dims, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
    def get_point_idx(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points_center = data_dict['points_center']
        xidx = torch.floor((points_center[:, 0] - self.pc_range[0][0])/self.voxel_size[0]).long().unsqueeze(1)
        yidx = torch.floor((points_center[:, 1] - self.pc_range[1][0])/self.voxel_size[1]).long().unsqueeze(1)
        zidx = torch.floor((points_center[:, 2] - self.pc_range[2][0])/self.voxel_size[2]).long().unsqueeze(1)
        center_coord_voxel = torch.cat([batch_idx.unsqueeze(1), xidx, yidx, zidx], dim=1)
        center_coord_unique, center_coord = torch.unique(center_coord_voxel, return_inverse=True, dim=0)
        return center_coord_voxel, center_coord, center_coord_unique
    def get_cluster_idx(self, points_cluster_idx):
        max_idx = points_cluster_idx.max() + 1
        return torch.arange(0, max_idx).to(points_cluster_idx)
        # return torch.unique(points_cluster_idx, dim=0)
    def cluster_comm(self, 
                     cluster_feature,
                     cluster_voxel_idx, 
                     cluster_idx,
                     points_feature, 
                     points_voxel_idx,
                     points_idx, 
                     voxel_unique
                     ):
        voxel_associate_unique = voxel_unique.clone()
        voxel_associate_unique[:, 0] = voxel_associate_unique[:, 0]/2
        voxel_associate_unique = voxel_associate_unique.long()
        associate_unique, voxel_associate_unique_idx = torch.unique(voxel_associate_unique, return_inverse=True, dim=0)
        cluster_associate_idx = voxel_associate_unique_idx[cluster_idx]
        points_associate_idx = voxel_associate_unique_idx[points_idx]
        
        cluster_feature_associate_all = torch.zeros((associate_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_associate_ = torch_scatter.scatter_mean(cluster_feature, cluster_associate_idx, dim=0)
        
        cluster_feature_associate_all[:cluster_feature_associate_.shape[0], :] = cluster_feature_associate_
        cluster_feature_associate = cluster_feature_associate_all[cluster_associate_idx]
        points_feature_associate = cluster_feature_associate_all[points_associate_idx]
        
        cluster_feature_all = torch.zeros((voxel_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_all[cluster_idx] = cluster_feature
        points_feature_cluster = cluster_feature_all[points_idx]
        
        associate_mask = ((points_feature_cluster - points_feature_associate).sum(dim=1)>0)
        if associate_mask.sum()<=1:
            return points_feature, associate_mask
        points_feature_associate = torch.cat([points_feature, points_feature_cluster, points_feature_associate], dim=1)
        points_feature_associate = self.pf_down(points_feature_associate[associate_mask])
        points_feature_ = points_feature.clone()
        points_feature_[associate_mask] = points_feature_associate.clone()
        return points_feature_, associate_mask
    def get_points_time_embedding(self, data_dict):
        timestamp = data_dict['points_cloud_timestamp']
        batch_idx = data_dict['batch_idx']
        time_other_idx = (batch_idx/2).long() * 2 + (1 - torch.fmod(batch_idx, 2)).long()
        time_delay = timestamp[batch_idx] - timestamp[time_other_idx]
        points_time_delay = get_position_embedding(time_delay, self.input_dims, self.time_pos_temperature)
        return points_time_delay
    def forward(self, data_dict, points_feature):
        points_time_delay = self.get_points_time_embedding(data_dict)
        points_time_delay = torch.zeros_like(points_time_delay)
        points_feature = points_feature + points_time_delay
        foreground_mask = data_dict['foreground_mask']
        points_center_fore = data_dict['points_center'][foreground_mask]
        points_feature_fore = points_feature[foreground_mask]
        points_time_delay_fore = points_time_delay[foreground_mask]
        points_voxel_idx, points_cluster_idx, points_voxel_unique = self.get_point_idx(data_dict)
        points_cluster_idx_fore = points_cluster_idx[foreground_mask]
        # points_cluster_idx, cluter_center_unq = self.get_cluster_idx(points_idx[foreground_mask])
        cluster_feature = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_time_delay = torch_scatter.scatter_mean(points_time_delay_fore, points_cluster_idx_fore, dim=0)
        cluster_idx = self.get_cluster_idx(points_cluster_idx_fore)
        cluster_voxel_idx = points_voxel_unique[cluster_idx]
        # 可以增加一个infra相对vehicle的time embedding vehicle测是反过来对infra的相对time embedding
        cluster_feature = self.cluster_proj(cluster_feature + cluster_time_delay)
        points_feature_new, points_associate_mask  = self.cluster_comm(cluster_feature,
                                               cluster_voxel_idx,
                                               cluster_idx,
                                               points_feature,
                                               points_voxel_idx,
                                               points_cluster_idx,
                                               points_voxel_unique)
        return points_feature_new
        
        
        
class InstanceCommCluster(nn.Module):
    def __init__(self, args):
        super(InstanceCommCluster, self).__init__()
        self.args = args
        self.voxel_size = self.args['voxel_size']
        self.pc_range = self.args['pc_range']
        self.input_dims = self.args['input_dims']
        self.time_pos_temperature = self.args['time_pos_temperature']
        self.pf_down = nn.Sequential(
            nn.Linear(self.input_dims*3, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
        self.cluster_proj = nn.Sequential(
            nn.Linear(self.input_dims, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
    def get_point_idx(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points_center = data_dict['points_center']
        xidx = torch.floor((points_center[:, 0] - self.pc_range[0][0])/self.voxel_size[0]).long().unsqueeze(1)
        yidx = torch.floor((points_center[:, 1] - self.pc_range[1][0])/self.voxel_size[1]).long().unsqueeze(1)
        zidx = torch.floor((points_center[:, 2] - self.pc_range[2][0])/self.voxel_size[2]).long().unsqueeze(1)
        center_coord_voxel = torch.cat([batch_idx.unsqueeze(1), xidx, yidx, zidx], dim=1)
        center_coord_unique, center_coord = torch.unique(center_coord_voxel, return_inverse=True, dim=0)
        return center_coord_voxel, center_coord, center_coord_unique
    def get_cluster_idx(self, points_cluster_idx):
        max_idx = points_cluster_idx.max() + 1
        return torch.arange(0, max_idx).to(points_cluster_idx)
        # return torch.unique(points_cluster_idx, dim=0)
    def cluster_comm(self, 
                     cluster_feature,
                     cluster_voxel_idx, 
                     cluster_idx,
                     points_feature, 
                     points_voxel_idx,
                     points_idx, 
                     voxel_unique
                     ):
        voxel_associate_unique = voxel_unique.clone()
        voxel_associate_unique[:, 0] = voxel_associate_unique[:, 0]/2
        voxel_associate_unique = voxel_associate_unique.long()
        associate_unique, voxel_associate_unique_idx = torch.unique(voxel_associate_unique, return_inverse=True, dim=0)
        cluster_associate_idx = voxel_associate_unique_idx[cluster_idx]
        points_associate_idx = voxel_associate_unique_idx[points_idx]
        
        cluster_feature_associate_all = torch.zeros((associate_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_associate_ = torch_scatter.scatter_mean(cluster_feature, cluster_associate_idx, dim=0)
        
        cluster_feature_associate_all[:cluster_feature_associate_.shape[0], :] = cluster_feature_associate_
        cluster_feature_associate = cluster_feature_associate_all[cluster_associate_idx]
        points_feature_associate = cluster_feature_associate_all[points_associate_idx]
        
        cluster_feature_all = torch.zeros((voxel_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_all[cluster_idx] = cluster_feature
        points_feature_cluster = cluster_feature_all[points_idx]
        
        associate_mask = ((points_feature_cluster - points_feature_associate).sum(dim=1)>0)
        if associate_mask.sum()<=1:
            return points_feature, associate_mask
        points_feature_associate = torch.cat([points_feature, points_feature_cluster, points_feature_associate], dim=1)
        points_feature_associate = self.pf_down(points_feature_associate[associate_mask])
        points_feature_ = points_feature.clone()
        points_feature_[associate_mask] = points_feature_associate.clone()
        return points_feature_, associate_mask
    def get_points_time_embedding(self, data_dict):
        timestamp = data_dict['points_cloud_timestamp']
        batch_idx = data_dict['batch_idx']
        time_other_idx = (batch_idx/2).long() * 2 + (1 - torch.fmod(batch_idx, 2)).long()
        time_delay = timestamp[batch_idx] - timestamp[time_other_idx]
        points_time_delay = get_position_embedding(time_delay, self.input_dims, self.time_pos_temperature)
        return points_time_delay
    def get_cluster_data(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_voxel_unique,
                         fg_mask,
                         ):
        output_data = {'filt_flag': True}
        # if fg_mask.sum()<=0:
        #     fg_mask = torch.ones_like(fg_mask)
            # B = points_voxel_unique[:, 0].max().item() + 1
            # for i in range(0, B):
            #     mask_b = (points_cluster_idx[:, 0] == i)
            #     fg_mask_b = fg_mask[mask_b]
            #     fg_mask_b[0] = True
            #     fg_mask[mask_b] = fg_mask_b
            # output_data['filt_flag'] = False
        points_feature_fore = points_feature[fg_mask]
        points_logits_fore = points_logits[fg_mask]
        points_cluster_idx_fore = points_cluster_idx[fg_mask]
        points_fore = points[fg_mask]
        points_proj_fore = points_proj[fg_mask]
        cluster_feature = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_logits = torch_scatter.scatter_mean(points_logits_fore, points_cluster_idx_fore, dim=0)
        cluster_center = torch_scatter.scatter_mean(points_fore, points_cluster_idx_fore, dim=0)
        cluster_center_proj = torch_scatter.scatter_mean(points_proj_fore, points_cluster_idx_fore, dim=0)
        cluster_idx = self.get_cluster_idx(points_cluster_idx_fore)
        cluster_idx_unique = torch.unique(points_cluster_idx_fore).long()
        cluster_mask = torch.zeros(cluster_idx.shape[0]).to(points_cluster_idx_fore).long()
        cluster_mask[cluster_idx_unique] = 1
        cluster_mask = cluster_mask>0
        if cluster_mask.sum()<=0:
            cluster_mask = torch.zeros(cluster_idx.shape[0]).to(points_cluster_idx_fore).long()
            cluster_mask = cluster_mask<=0
        cluster_batchidx = points_voxel_unique[cluster_idx][:, 0]
        output_data['points'] = cluster_center[cluster_mask]
        output_data['points_proj'] = cluster_center_proj[cluster_mask]
        output_data['points_feature'] = cluster_feature[cluster_mask]
        output_data['logits'] = cluster_logits[cluster_mask]
        output_data['batch_idx'] = cluster_batchidx[cluster_mask]
        output_data['batch_size'] = cluster_batchidx[cluster_mask].max().item() + 1
        return output_data
    def forward(self, data_dict, points_feature):
        points_time_delay = self.get_points_time_embedding(data_dict)
        points_time_delay = torch.zeros_like(points_time_delay)
        points_feature = points_feature + points_time_delay
        foreground_mask = data_dict['foreground_mask']
        points_center_fore = data_dict['points_center'][foreground_mask]
        points_fore = data_dict['points'][foreground_mask]
        points_feature_fore = points_feature[foreground_mask]
        points_time_delay_fore = points_time_delay[foreground_mask]
        points_voxel_idx, points_cluster_idx, points_voxel_unique = self.get_point_idx(data_dict)
        points_cluster_idx_fore = points_cluster_idx[foreground_mask]
        # points_cluster_idx, cluter_center_unq = self.get_cluster_idx(points_idx[foreground_mask])
        cluster_feature = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_time_delay = torch_scatter.scatter_mean(points_time_delay_fore, points_cluster_idx_fore, dim=0)
        cluster_idx = self.get_cluster_idx(points_cluster_idx_fore)
        cluster_voxel_idx = points_voxel_unique[cluster_idx]
        # 可以增加一个infra相对vehicle的time embedding vehicle测是反过来对infra的相对time embedding
        cluster_feature = self.cluster_proj(cluster_feature + cluster_time_delay)
        points_feature_new, points_associate_mask  = self.cluster_comm(cluster_feature,
                                               cluster_voxel_idx,
                                               cluster_idx,
                                               points_feature,
                                               points_voxel_idx,
                                               points_cluster_idx,
                                               points_voxel_unique)
        output_data_cluster = self.get_cluster_data(points=data_dict['points'],
                                                    points_proj=data_dict['points_proj'],
                                                    points_feature=data_dict['points_feature'],
                                                    points_logits=data_dict['logits'],
                                                    points_cluster_idx=points_cluster_idx,
                                                    points_voxel_unique=points_voxel_unique,
                                                    fg_mask=data_dict['foreground_mask'])
        return points_feature_new, output_data_cluster
        
        

class InstanceCommClusterPoints(nn.Module):
    def __init__(self, args):
        super(InstanceCommClusterPoints, self).__init__()
        self.args = args
        self.voxel_size = self.args['voxel_size']
        self.pc_range = self.args['pc_range']
        self.input_dims = self.args['input_dims']
        self.time_pos_temperature = self.args['time_pos_temperature']
        self.compression_dim = self.args['compression_dim'] if 'compression_dim' in self.args else 64
        if 'training_all_fg' not in self.args:
            self.training_all_fg = False
        else:
            self.training_all_fg = self.args['training_all_fg']
        self.pf_down = nn.Sequential(
            nn.Linear((self.input_dims+20+4+3)*2 - self.input_dims, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
        self.pf_commpresion = nn.Sequential(
            nn.Linear(self.input_dims, self.compression_dim),
            nn.BatchNorm1d(self.compression_dim),
            nn.LeakyReLU(0.1, True),
        )
        self.pf_up_compresion = nn.Sequential(
            nn.Linear(self.compression_dim, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
        print(self.voxel_size)
        print(self.training_all_fg)
        # self.cluster_proj = nn.Sequential(
        #     nn.Linear(self.input_dims, self.input_dims),
        #     nn.BatchNorm1d(self.input_dims),
        #     nn.LeakyReLU(0.1, True),
        # )
    def get_point_idx(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points_center = data_dict['points_center']
        xidx = torch.floor((points_center[:, 0] - self.pc_range[0][0])/self.voxel_size[0]).long().unsqueeze(1)
        yidx = torch.floor((points_center[:, 1] - self.pc_range[1][0])/self.voxel_size[1]).long().unsqueeze(1)
        zidx = torch.floor((points_center[:, 2] - self.pc_range[2][0])/self.voxel_size[2]).long().unsqueeze(1)
        center_coord_voxel = torch.cat([batch_idx.unsqueeze(1), xidx, yidx, zidx], dim=1)
        center_coord_unique, center_coord = torch.unique(center_coord_voxel, return_inverse=True, dim=0)
        return center_coord_voxel, center_coord, center_coord_unique
    def get_cluster_idx(self, points_cluster_idx):
        max_idx = points_cluster_idx.max() + 1
        return torch.arange(0, max_idx).to(points_cluster_idx)
        # return torch.unique(points_cluster_idx, dim=0)
    def cluster_comm(self, 
                     cluster_feature,
                     cluster_voxel_idx, 
                     cluster_idx,
                     points_feature, 
                     points_voxel_idx,
                     points_idx, 
                     voxel_unique
                     ):
        voxel_associate_unique = voxel_unique.clone()
        voxel_associate_unique[:, 0] = voxel_associate_unique[:, 0]/2
        voxel_associate_unique = voxel_associate_unique.long()
        associate_unique, voxel_associate_unique_idx = torch.unique(voxel_associate_unique, return_inverse=True, dim=0)
        cluster_associate_idx = voxel_associate_unique_idx[cluster_idx]
        points_associate_idx = voxel_associate_unique_idx[points_idx]
        
        cluster_feature_associate_all = torch.zeros((associate_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_associate_ = torch_scatter.scatter_mean(cluster_feature, cluster_associate_idx, dim=0)
        
        cluster_feature_associate_all[:cluster_feature_associate_.shape[0], :] = cluster_feature_associate_
        cluster_feature_associate = cluster_feature_associate_all[cluster_associate_idx]
        points_feature_associate = cluster_feature_associate_all[points_associate_idx]
        
        cluster_feature_all = torch.zeros((voxel_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_all[cluster_idx] = cluster_feature
        points_feature_cluster = cluster_feature_all[points_idx]
        
        associate_mask = ((points_feature_cluster - points_feature_associate).sum(dim=1)>0)
        if associate_mask.sum()<=1:
            return points_feature, associate_mask
        points_feature_associate = torch.cat([points_feature, points_feature_cluster, points_feature_associate], dim=1)
        points_feature_associate = self.pf_down(points_feature_associate[associate_mask])
        points_feature_ = points_feature.clone()
        points_feature_[associate_mask] = points_feature_associate.clone()
        return points_feature_, associate_mask
    def get_points_time_embedding(self, data_dict):
        timestamp = data_dict['points_cloud_timestamp']
        batch_idx = data_dict['batch_idx']
        time_other_idx = (batch_idx/2).long() * 2 + (1 - torch.fmod(batch_idx, 2)).long()
        time_delay = timestamp[batch_idx] - timestamp[time_other_idx]
        points_time_delay = get_position_embedding(time_delay, self.input_dims, self.time_pos_temperature)
        return points_time_delay
    def get_cluster_data_vi(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_voxel_unique,
                         fg_mask,
                         ):
        # if fg_mask.sum()<=0:
        #     fg_mask = torch.ones_like(fg_mask)
            # B = points_voxel_unique[:, 0].max().item() + 1
            # for i in range(0, B):
            #     mask_b = (points_cluster_idx[:, 0] == i)
            #     fg_mask_b = fg_mask[mask_b]
            #     fg_mask_b[0] = True
            #     fg_mask[mask_b] = fg_mask_b
            # output_data['filt_flag'] = False
        points_feature_fore = points_feature[fg_mask]
        points_logits_fore = points_logits[fg_mask]
        points_cluster_idx_fore = points_cluster_idx[fg_mask]
        points_fore = points[fg_mask]
        points_proj_fore = points_proj[fg_mask]
        
        bg_mask = ~fg_mask
        points_feature_bg = points_feature[bg_mask]
        points_logits_bg = points_logits[bg_mask]
        points_cluster_idx_bg = points_cluster_idx[bg_mask]
        points_bg = points[bg_mask]
        points_proj_bg = points_proj[bg_mask]
        
        cluster_feature_fg = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_logits_fg = torch_scatter.scatter_mean(points_logits_fore, points_cluster_idx_fore, dim=0)
        cluster_center_fg = torch_scatter.scatter_mean(points_fore, points_cluster_idx_fore, dim=0)
        cluster_center_proj_fg = torch_scatter.scatter_mean(points_proj_fore, points_cluster_idx_fore, dim=0)
        
        cluster_feature_bg = torch_scatter.scatter_mean(points_feature_bg, points_cluster_idx_bg, dim=0)
        cluster_logits_bg = torch_scatter.scatter_mean(points_logits_bg, points_cluster_idx_bg, dim=0)
        cluster_center_bg = torch_scatter.scatter_mean(points_bg, points_cluster_idx_bg, dim=0)
        cluster_center_proj_bg = torch_scatter.scatter_mean(points_proj_bg, points_cluster_idx_bg, dim=0)
        
        points_cluster_fg = cluster_feature_fg[points_cluster_idx_fore]
        points_cluster_fg_logits = cluster_logits_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center = cluster_center_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center_proj = cluster_center_proj_fg[points_cluster_idx_fore]
        points_feature_fg = torch.cat([points_feature_fore, 
                                        points_logits_fore, 
                                        points_fore,
                                        points_proj_fore,
                                        points_cluster_fg,
                                        points_cluster_fg_logits,
                                        poitns_cluster_fg_center,
                                        poitns_cluster_fg_center_proj], dim=1)
        
        points_cluster_bg = cluster_feature_bg[points_cluster_idx_bg]
        points_cluster_bg_logits = cluster_logits_bg[points_cluster_idx_bg]
        poitns_cluster_bg_center = cluster_center_bg[points_cluster_idx_bg]
        poitns_cluster_bg_center_proj = cluster_center_proj_bg[points_cluster_idx_bg]
        points_feature_bg = torch.cat([points_feature_bg, 
                                        points_logits_bg, 
                                        points_bg,
                                        points_proj_bg,
                                        points_cluster_bg,
                                        points_cluster_bg_logits,
                                        poitns_cluster_bg_center,
                                        poitns_cluster_bg_center_proj], dim=1)
        points_feature_new = torch.zeros_like(points_feature)
        points_feature_new[fg_mask] = points_feature_fg
        points_feature_new[bg_mask] = points_feature_bg
        points_feature_new = self.pf_down(points_feature_new)
        
        output_data_v = {'filt_flag': False}
        output_data_v['batch_size'] = output_data_v['batch_idx'].max().item() + 1
        output_data_v['points'] = []
        output_data_v['points_proj'] = []
        output_data_v['points_feature'] = []
        output_data_v['logits'] = []
        output_data_v['batch_idx'] = []
        batch_idx = points_voxel_unique[points_cluster_idx][:, 0]
        for b_idx in range(0, output_data_v['batch_size']):
            mask_b = (batch_idx == b_idx)
            if b_idx % 2 == 1:
                mask_b = mask_b & fg_mask
            points_b = points[mask_b]
            points_proj_b = points_proj[mask_b]
            points_feature_new_b = points_feature_new[mask_b]
            points_logits_b = points_logits[mask_b]
            batch_idx_b = batch_idx[mask_b]
            output_data_v['points'].append(points_b)
            output_data_v['points_proj'].append(points_proj_b)
            output_data_v['points_feature'].append(points_feature_new_b)
            output_data_v['logits'].append(points_logits_b)
            output_data_v['batch_idx'].append(batch_idx_b)
        output_data_v['points'] = torch.cat(output_data_v['points'], dim=0)
        output_data_v['points_proj'] = torch.cat(output_data_v['points_proj'], dim=0)
        output_data_v['points_feature'] = torch.cat(output_data_v['points_feature'], dim=0)
        output_data_v['logits'] = torch.cat(output_data_v['logits'], dim=0)
        output_data_v['batch_idx'] = torch.cat(output_data_v['batch_idx'], dim=0)
        
        output_data_i = {'filt_flag': False}
        output_data_i['batch_size'] = output_data_i['batch_idx'].max().item() + 1
        output_data_i['points'] = []
        output_data_i['points_proj'] = []
        output_data_i['points_feature'] = []
        output_data_i['logits'] = []
        output_data_i['batch_idx'] = []
        batch_idx = points_voxel_unique[points_cluster_idx][:, 0]
        for b_idx in range(0, output_data_i['batch_size']):
            mask_b = (batch_idx == b_idx)
            if b_idx % 2 == 0:
                mask_b = mask_b & fg_mask
            points_b = points[mask_b]
            points_proj_b = points_proj[mask_b]
            points_feature_new_b = points_feature_new[mask_b]
            points_logits_b = points_logits[mask_b]
            batch_idx_b = batch_idx[mask_b]
            output_data_i['points'].append(points_b)
            output_data_i['points_proj'].append(points_proj_b)
            output_data_i['points_feature'].append(points_feature_new_b)
            output_data_i['logits'].append(points_logits_b)
            output_data_i['batch_idx'].append(batch_idx_b)
        output_data_i['points'] = torch.cat(output_data_i['points'], dim=0)
        output_data_i['points_proj'] = torch.cat(output_data_i['points_proj'], dim=0)
        output_data_i['points_feature'] = torch.cat(output_data_i['points_feature'], dim=0)
        output_data_i['logits'] = torch.cat(output_data_i['logits'], dim=0)
        output_data_i['batch_idx'] = torch.cat(output_data_i['batch_idx'], dim=0)
        return output_data_v, output_data_i
    def get_cluster_data(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_voxel_unique,
                         fg_mask,
                         ):
        # if fg_mask.sum()<=0:
        #     fg_mask = torch.ones_like(fg_mask)
            # B = points_voxel_unique[:, 0].max().item() + 1
            # for i in range(0, B):
            #     mask_b = (points_cluster_idx[:, 0] == i)
            #     fg_mask_b = fg_mask[mask_b]
            #     fg_mask_b[0] = True
            #     fg_mask[mask_b] = fg_mask_b
            # output_data['filt_flag'] = False
        if self.training and self.training_all_fg: # 这个是关键步骤需要在训练的时候所有feature一起训练才能增加它的泛化性能
            fg_mask = torch.ones_like(fg_mask)
        points_feature_fore = points_feature[fg_mask]
        points_logits_fore = points_logits[fg_mask]
        points_cluster_idx_fore = points_cluster_idx[fg_mask]
        points_fore = points[fg_mask]
        points_proj_fore = points_proj[fg_mask]
        
        cluster_feature_fg = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_logits_fg = torch_scatter.scatter_mean(points_logits_fore, points_cluster_idx_fore, dim=0)
        cluster_center_fg = torch_scatter.scatter_mean(points_fore, points_cluster_idx_fore, dim=0)
        cluster_center_proj_fg = torch_scatter.scatter_mean(points_proj_fore, points_cluster_idx_fore, dim=0)
        
        points_cluster_fg = cluster_feature_fg[points_cluster_idx_fore]
        points_cluster_fg_logits = cluster_logits_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center = cluster_center_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center_proj = cluster_center_proj_fg[points_cluster_idx_fore]
        # points_feature_fg = torch.cat([ points_logits_fore, 
        #                                 points_fore,
        #                                 points_proj_fore,
        #                                 points_cluster_fg,
        #                                 points_cluster_fg_logits,
        #                                 poitns_cluster_fg_center,
        #                                 poitns_cluster_fg_center_proj], dim=1)
        
        points_feature_fg = torch.cat([ points_cluster_fg_logits, 
                                        poitns_cluster_fg_center,
                                        poitns_cluster_fg_center_proj,
                                        points_cluster_fg,
                                        points_cluster_fg_logits,
                                        poitns_cluster_fg_center,
                                        poitns_cluster_fg_center_proj], dim=1)
        
        # points_feature_new = torch.zeros_like(points_feature)
        points_feature_new = points_feature_fg
        points_feature_new = self.pf_down(points_feature_new)
        points_feature_new = self.pf_commpresion(points_feature_new)
        comm_channel = points_feature_new.shape[1]
        points_feature_new = self.pf_up_compresion(points_feature_new)
        output_data = {'filt_flag': False}
        output_data['points'] = points_fore #points
        output_data['points_proj'] = points_proj_fore # points_proj
        output_data['points_feature'] = points_feature_new
        output_data['points_cluster_center'] = poitns_cluster_fg_center
        output_data['points_cluster_logits'] = points_cluster_fg_logits
        output_data['logits'] = points_logits_fore # points_logits
        output_data['batch_idx'] = points_voxel_unique[points_cluster_idx_fore][:, 0] # points_voxel_unique[points_cluster_idx][:, 0]
        output_data['batch_size'] = output_data['batch_idx'].max().item() + 1
        points_cluster_idx_fore_v = []
        for b_idx in range(0, output_data['batch_size']):
            if b_idx % 2 == 1:
                continue
            mask_b = (output_data['batch_idx'] == b_idx)
            points_cluster_idx_foreb = points_cluster_idx_fore[mask_b]
            points_cluster_idx_fore_v.append(points_cluster_idx_foreb)
        points_cluster_idx_fore_v = torch.concat(points_cluster_idx_fore_v, dim=0)
        cluster_v_unique = torch.unique(points_cluster_idx_fore_v, dim=0)
        if self.training is False:
            print(cluster_v_unique.shape)
            print(comm_channel)
        output_data['communication_rates'] = torch.tensor(1).to(points_fore.device) * cluster_v_unique.shape[0] * comm_channel # * points_feature_new.shape[1]
        output_data['communication_bytes'] = torch.tensor(1).to(points_fore.device) * cluster_v_unique.shape[0] * comm_channel
        return output_data
    def forward(self, data_dict):
        points_voxel_idx, points_cluster_idx, points_voxel_unique = self.get_point_idx(data_dict)
        output_data_cluster = self.get_cluster_data(points=data_dict['points'],
                                                    points_proj=data_dict['points_proj'],
                                                    points_feature=data_dict['points_feature'],
                                                    points_logits=data_dict['logits'],
                                                    points_cluster_idx=points_cluster_idx,
                                                    points_voxel_unique=points_voxel_unique,
                                                    fg_mask=data_dict['foreground_mask'])
        return output_data_cluster
    
    
class InstanceCommClusterPointsFull(nn.Module):
    def __init__(self, args):
        super(InstanceCommClusterPointsFull, self).__init__()
        self.args = args
        self.voxel_size = self.args['voxel_size']
        self.pc_range = self.args['pc_range']
        self.input_dims = self.args['input_dims']
        self.time_pos_temperature = self.args['time_pos_temperature']
        self.pf_down = nn.Sequential(
            nn.Linear((self.input_dims+20+4+3)*2 - self.input_dims, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.LeakyReLU(0.1, True),
        )
        # self.cluster_proj = nn.Sequential(
        #     nn.Linear(self.input_dims, self.input_dims),
        #     nn.BatchNorm1d(self.input_dims),
        #     nn.LeakyReLU(0.1, True),
        # )
    def get_point_idx(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points_center = data_dict['points_center']
        xidx = torch.floor((points_center[:, 0] - self.pc_range[0][0])/self.voxel_size[0]).long().unsqueeze(1)
        yidx = torch.floor((points_center[:, 1] - self.pc_range[1][0])/self.voxel_size[1]).long().unsqueeze(1)
        zidx = torch.floor((points_center[:, 2] - self.pc_range[2][0])/self.voxel_size[2]).long().unsqueeze(1)
        center_coord_voxel = torch.cat([batch_idx.unsqueeze(1), xidx, yidx, zidx], dim=1)
        center_coord_unique, center_coord = torch.unique(center_coord_voxel, return_inverse=True, dim=0)
        return center_coord_voxel, center_coord, center_coord_unique
    def get_cluster_idx(self, points_cluster_idx):
        max_idx = points_cluster_idx.max() + 1
        return torch.arange(0, max_idx).to(points_cluster_idx)
        # return torch.unique(points_cluster_idx, dim=0)
    def cluster_comm(self, 
                     cluster_feature,
                     cluster_voxel_idx, 
                     cluster_idx,
                     points_feature, 
                     points_voxel_idx,
                     points_idx, 
                     voxel_unique
                     ):
        voxel_associate_unique = voxel_unique.clone()
        voxel_associate_unique[:, 0] = voxel_associate_unique[:, 0]/2
        voxel_associate_unique = voxel_associate_unique.long()
        associate_unique, voxel_associate_unique_idx = torch.unique(voxel_associate_unique, return_inverse=True, dim=0)
        cluster_associate_idx = voxel_associate_unique_idx[cluster_idx]
        points_associate_idx = voxel_associate_unique_idx[points_idx]
        
        cluster_feature_associate_all = torch.zeros((associate_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_associate_ = torch_scatter.scatter_mean(cluster_feature, cluster_associate_idx, dim=0)
        
        cluster_feature_associate_all[:cluster_feature_associate_.shape[0], :] = cluster_feature_associate_
        cluster_feature_associate = cluster_feature_associate_all[cluster_associate_idx]
        points_feature_associate = cluster_feature_associate_all[points_associate_idx]
        
        cluster_feature_all = torch.zeros((voxel_unique.shape[0], cluster_feature.shape[1])).to(cluster_feature)
        cluster_feature_all[cluster_idx] = cluster_feature
        points_feature_cluster = cluster_feature_all[points_idx]
        
        associate_mask = ((points_feature_cluster - points_feature_associate).sum(dim=1)>0)
        if associate_mask.sum()<=1:
            return points_feature, associate_mask
        points_feature_associate = torch.cat([points_feature, points_feature_cluster, points_feature_associate], dim=1)
        points_feature_associate = self.pf_down(points_feature_associate[associate_mask])
        points_feature_ = points_feature.clone()
        points_feature_[associate_mask] = points_feature_associate.clone()
        return points_feature_, associate_mask
    def get_points_time_embedding(self, data_dict):
        timestamp = data_dict['points_cloud_timestamp']
        batch_idx = data_dict['batch_idx']
        time_other_idx = (batch_idx/2).long() * 2 + (1 - torch.fmod(batch_idx, 2)).long()
        time_delay = timestamp[batch_idx] - timestamp[time_other_idx]
        points_time_delay = get_position_embedding(time_delay, self.input_dims, self.time_pos_temperature)
        return points_time_delay
    def get_cluster_data_vi(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_voxel_unique,
                         fg_mask,
                         ):
        # if fg_mask.sum()<=0:
        #     fg_mask = torch.ones_like(fg_mask)
            # B = points_voxel_unique[:, 0].max().item() + 1
            # for i in range(0, B):
            #     mask_b = (points_cluster_idx[:, 0] == i)
            #     fg_mask_b = fg_mask[mask_b]
            #     fg_mask_b[0] = True
            #     fg_mask[mask_b] = fg_mask_b
            # output_data['filt_flag'] = False
        points_feature_fore = points_feature[fg_mask]
        points_logits_fore = points_logits[fg_mask]
        points_cluster_idx_fore = points_cluster_idx[fg_mask]
        points_fore = points[fg_mask]
        points_proj_fore = points_proj[fg_mask]
        
        bg_mask = ~fg_mask
        points_feature_bg = points_feature[bg_mask]
        points_logits_bg = points_logits[bg_mask]
        points_cluster_idx_bg = points_cluster_idx[bg_mask]
        points_bg = points[bg_mask]
        points_proj_bg = points_proj[bg_mask]
        
        cluster_feature_fg = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_logits_fg = torch_scatter.scatter_mean(points_logits_fore, points_cluster_idx_fore, dim=0)
        cluster_center_fg = torch_scatter.scatter_mean(points_fore, points_cluster_idx_fore, dim=0)
        cluster_center_proj_fg = torch_scatter.scatter_mean(points_proj_fore, points_cluster_idx_fore, dim=0)
        
        cluster_feature_bg = torch_scatter.scatter_mean(points_feature_bg, points_cluster_idx_bg, dim=0)
        cluster_logits_bg = torch_scatter.scatter_mean(points_logits_bg, points_cluster_idx_bg, dim=0)
        cluster_center_bg = torch_scatter.scatter_mean(points_bg, points_cluster_idx_bg, dim=0)
        cluster_center_proj_bg = torch_scatter.scatter_mean(points_proj_bg, points_cluster_idx_bg, dim=0)
        
        points_cluster_fg = cluster_feature_fg[points_cluster_idx_fore]
        points_cluster_fg_logits = cluster_logits_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center = cluster_center_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center_proj = cluster_center_proj_fg[points_cluster_idx_fore]
        points_feature_fg = torch.cat([points_feature_fore, 
                                        points_logits_fore, 
                                        points_fore,
                                        points_proj_fore,
                                        points_cluster_fg,
                                        points_cluster_fg_logits,
                                        poitns_cluster_fg_center,
                                        poitns_cluster_fg_center_proj], dim=1)
        
        points_cluster_bg = cluster_feature_bg[points_cluster_idx_bg]
        points_cluster_bg_logits = cluster_logits_bg[points_cluster_idx_bg]
        poitns_cluster_bg_center = cluster_center_bg[points_cluster_idx_bg]
        poitns_cluster_bg_center_proj = cluster_center_proj_bg[points_cluster_idx_bg]
        points_feature_bg = torch.cat([points_feature_bg, 
                                        points_logits_bg, 
                                        points_bg,
                                        points_proj_bg,
                                        points_cluster_bg,
                                        points_cluster_bg_logits,
                                        poitns_cluster_bg_center,
                                        poitns_cluster_bg_center_proj], dim=1)
        points_feature_new = torch.zeros_like(points_feature)
        points_feature_new[fg_mask] = points_feature_fg
        points_feature_new[bg_mask] = points_feature_bg
        points_feature_new = self.pf_down(points_feature_new)
        
        output_data_v = {'filt_flag': False}
        output_data_v['batch_size'] = output_data_v['batch_idx'].max().item() + 1
        output_data_v['points'] = []
        output_data_v['points_proj'] = []
        output_data_v['points_feature'] = []
        output_data_v['logits'] = []
        output_data_v['batch_idx'] = []
        batch_idx = points_voxel_unique[points_cluster_idx][:, 0]
        for b_idx in range(0, output_data_v['batch_size']):
            mask_b = (batch_idx == b_idx)
            if b_idx % 2 == 1:
                mask_b = mask_b & fg_mask
            points_b = points[mask_b]
            points_proj_b = points_proj[mask_b]
            points_feature_new_b = points_feature_new[mask_b]
            points_logits_b = points_logits[mask_b]
            batch_idx_b = batch_idx[mask_b]
            output_data_v['points'].append(points_b)
            output_data_v['points_proj'].append(points_proj_b)
            output_data_v['points_feature'].append(points_feature_new_b)
            output_data_v['logits'].append(points_logits_b)
            output_data_v['batch_idx'].append(batch_idx_b)
        output_data_v['points'] = torch.cat(output_data_v['points'], dim=0)
        output_data_v['points_proj'] = torch.cat(output_data_v['points_proj'], dim=0)
        output_data_v['points_feature'] = torch.cat(output_data_v['points_feature'], dim=0)
        output_data_v['logits'] = torch.cat(output_data_v['logits'], dim=0)
        output_data_v['batch_idx'] = torch.cat(output_data_v['batch_idx'], dim=0)
        
        output_data_i = {'filt_flag': False}
        output_data_i['batch_size'] = output_data_i['batch_idx'].max().item() + 1
        output_data_i['points'] = []
        output_data_i['points_proj'] = []
        output_data_i['points_feature'] = []
        output_data_i['logits'] = []
        output_data_i['batch_idx'] = []
        batch_idx = points_voxel_unique[points_cluster_idx][:, 0]
        for b_idx in range(0, output_data_i['batch_size']):
            mask_b = (batch_idx == b_idx)
            if b_idx % 2 == 0:
                mask_b = mask_b & fg_mask
            points_b = points[mask_b]
            points_proj_b = points_proj[mask_b]
            points_feature_new_b = points_feature_new[mask_b]
            points_logits_b = points_logits[mask_b]
            batch_idx_b = batch_idx[mask_b]
            output_data_i['points'].append(points_b)
            output_data_i['points_proj'].append(points_proj_b)
            output_data_i['points_feature'].append(points_feature_new_b)
            output_data_i['logits'].append(points_logits_b)
            output_data_i['batch_idx'].append(batch_idx_b)
        output_data_i['points'] = torch.cat(output_data_i['points'], dim=0)
        output_data_i['points_proj'] = torch.cat(output_data_i['points_proj'], dim=0)
        output_data_i['points_feature'] = torch.cat(output_data_i['points_feature'], dim=0)
        output_data_i['logits'] = torch.cat(output_data_i['logits'], dim=0)
        output_data_i['batch_idx'] = torch.cat(output_data_i['batch_idx'], dim=0)
        return output_data_v, output_data_i
    def get_cluster_data(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_voxel_unique,
                         fg_mask,
                         ):
        # if fg_mask.sum()<=0:
        #     fg_mask = torch.ones_like(fg_mask)
            # B = points_voxel_unique[:, 0].max().item() + 1
            # for i in range(0, B):
            #     mask_b = (points_cluster_idx[:, 0] == i)
            #     fg_mask_b = fg_mask[mask_b]
            #     fg_mask_b[0] = True
            #     fg_mask[mask_b] = fg_mask_b
            # output_data['filt_flag'] = False
        fg_mask = torch.ones_like(fg_mask)
        points_feature_fore = points_feature[fg_mask]
        points_logits_fore = points_logits[fg_mask]
        points_cluster_idx_fore = points_cluster_idx[fg_mask]
        points_fore = points[fg_mask]
        points_proj_fore = points_proj[fg_mask]
        
        cluster_feature_fg = torch_scatter.scatter_mean(points_feature_fore, points_cluster_idx_fore, dim=0)
        cluster_logits_fg = torch_scatter.scatter_mean(points_logits_fore, points_cluster_idx_fore, dim=0)
        cluster_center_fg = torch_scatter.scatter_mean(points_fore, points_cluster_idx_fore, dim=0)
        cluster_center_proj_fg = torch_scatter.scatter_mean(points_proj_fore, points_cluster_idx_fore, dim=0)
        
        points_cluster_fg = cluster_feature_fg[points_cluster_idx_fore]
        points_cluster_fg_logits = cluster_logits_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center = cluster_center_fg[points_cluster_idx_fore]
        poitns_cluster_fg_center_proj = cluster_center_proj_fg[points_cluster_idx_fore]
        points_feature_fg = torch.cat([ points_logits_fore, 
                                        points_fore,
                                        points_proj_fore,
                                        points_feature_fore,
                                        points_logits_fore,
                                        points_fore,
                                        points_proj_fore], dim=1)
        
        # points_feature_new = torch.zeros_like(points_feature)
        points_feature_new = points_feature_fg
        points_feature_new = self.pf_down(points_feature_new)
        
        output_data = {'filt_flag': False}
        output_data['points'] = points
        output_data['points_proj'] = points_proj
        output_data['points_feature'] = points_feature_new
        output_data['logits'] = points_logits
        output_data['batch_idx'] = points_voxel_unique[points_cluster_idx][:, 0]
        output_data['batch_size'] = output_data['batch_idx'].max().item() + 1
        return output_data
    def forward(self, data_dict):
        points_voxel_idx, points_cluster_idx, points_voxel_unique = self.get_point_idx(data_dict)
        output_data_cluster = self.get_cluster_data(points=data_dict['points'],
                                                                             points_proj=data_dict['points_proj'],
                                                                             points_feature=data_dict['points_feature'],
                                                                             points_logits=data_dict['logits'],
                                                                             points_cluster_idx=points_cluster_idx,
                                                                             points_voxel_unique=points_voxel_unique,
                                                                             fg_mask=data_dict['foreground_mask'])
        return output_data_cluster