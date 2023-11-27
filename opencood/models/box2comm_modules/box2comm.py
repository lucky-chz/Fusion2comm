import torch
import torch.nn as nn
from copy import deepcopy
import torch_scatter
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

class RPNPostProcess(nn.Module):
    def __init__(self, args):
        super(RPNPostProcess, self).__init__(*args, **kwargs)
        self.params = deepcopy(args)
    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)?? should be (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d
    def get_boxes_3d(self, anchor_box, prob, reg):
        # classification probability
        prob = F.sigmoid(prob.permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)

        # convert regression map back to bounding box
        batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
        mask = \
            torch.gt(prob, self.params['target_args']['score_threshold_single'])
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

        # during validation/testing, the batch size should be 1
        assert batch_box3d.shape[0] == 1
        boxes3d = torch.masked_select(batch_box3d[0],
                                        mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])
        return boxes3d, scores
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            scores = torch.masked_select(prob[0], mask[0])
            boxes3d_single_v, scores_single_v = self.get_boxes_3d(anchor_box, 
                                                                  output_dict[cav_id]['psm_single_v'],
                                                                  output_dict[cav_id]['rm_single_v'])
            boxes3d_single_i, scores_single_i = self.get_boxes_3d(anchor_box, 
                                                                  output_dict[cav_id]['psm_single_i'],
                                                                  output_dict[cav_id]['rm_single_i'])
            boxes3d_single_i_corner = box_utils.boxes_to_corners_3d(boxes3d_single_i, order=self.params['order'])
            projected_boxes3d_single_i = box_utils.project_box3d(boxes3d_single_i_corner.float(), data_dict['ego']['pairwise_t_matrix'][0, 1, 0, :, :].float())
            projected_boxes3d_single_i = box_utils.corner_to_center_torch(projected_boxes3d_single_i, self.params['order'])
            boxes3d_single_vi = torch.cat([boxes3d_single_v, projected_boxes3d_single_i], dim=0)
            scores_vi = torch.cat([scores_single_v, scores_single_i], dim=0)
            boxes3d = boxes3d_single_vi
            scores = scores_vi
            
            # adding dir classifier
            if 'dm' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']


                dm  = output_dict[cav_id]['dm'] # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]
            
            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner.float(),
                                            transformation_matrix.float())
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # # remove large bbx
        # keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        # keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        # keep_index = torch.logical_and(keep_index_1, keep_index_2)

        # pred_box3d_tensor = pred_box3d_tensor[keep_index]
        # scores = scores[keep_index]
        
        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # # filter out the prediction out of the range.
        # mask = \
        #     box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
        # pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        # scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]
        return pred_box3d_tensor, scores
    
    def forward(self, data_dict, output_dict):
        batch_size = output_dict['batch_size']
        points = data_dict['points']
        batch_idx = data_dict['batch_idx']
        point_indices = []
        indices_base = 0
        boxbatch_idx = []
        for b_idx in range(0, batch_size):
            output_dict_b = {
                'ego':{
                    'psm_single_v': output_dict['psm_single_v'][b_idx, :, :, :].unsqueeze(0),
                    'psm_single_i': output_dict['psm_single_i'][b_idx, :, :, :].unsqueeze(0),
                    'rm_single_v': output_dict['rm_single_v'][b_idx, :, :, :].unsqueeze(0),
                    'rm_single_i': output_dict['rm_single_i'][b_idx, :, :, :].unsqueeze(0),
                }
            }
            pred_box3d_tensor, scores = self.post_process(data_dict, output_dict_b)
            mask_b = (batch_idx == b_idx)
            points_b = points[mask_b]
            pred_boxes_b = pred_box3d_tensor[:, 0:7].unsqueeze(0)
            point_indices_b = points_in_boxes_gpu(points_b, pred_boxes_b)[0]
            num_box_p = point_indices_b.max().item() + 1
            box_batch_idx.append(torch.ones((num_box_p, )) * b_idx)
            fg_mask_b = (point_indices_b>=0)
            point_indices_b[fg_mask_b] = point_indices_b[fg_mask_b] + indices_base
            indices_base = point_indices_b.max().item() + 1
            point_indices.append(point_indices_b)
            
        point_indices = torch.cat(point_indices, dim=0)
        box_batch_idx = torch.cat(box_batch_idx, dim=0).to(point_indices)
        if self.training:
            # `data_dict['instance_label']` is used to filter out points that belong
            # to instances. It is a binary mask where `True` indicates that a point
            # belongs to an instance and `False` indicates that a point does not
            # belong to an instance. This mask is used in the `Box2Comm` module to
            # determine which points should be considered for further processing.
            instance_label = data_dict['instance_label']
            inst_unq, inst_inv = torch.unique(instance_label, return_inverse=True, dim=0)
            instance_mask = (data_dict['instance_label']>=0)
            instance_label = inst_inv
            instance_label[~instance_mask] = -1
            box_mask = (point_indices>=0)
            unique_mask = (instance_mask == True) & (box_mask == False)
            indices_base = point_indices_b.max().item() + 1
            point_indices[unique_mask] = instance_label[unique_mask] + indices_base
            
            num_inst = instance_label[unique_mask].max().item() + 1
            inst_batch_idx_add = -torch.ones((num_inst,))
            box_batch_idx = torch.cat([box_batch_idx, inst_batch_idx_add], dim=0)
            inst_label_uni = instance_label[unique_mask]
            inst_batch_idx_ori = batch_idx[unique_mask]
            box_batch_idx[point_indices[unique_mask]] = inst_batch_idx_ori
            
        data_dict['point_box_indices'] = point_indices
        data_dict['box_batch_idx'] = box_batch_idx
        return data_dict

class Box2Comm(nn.Module):
    def __init__(self, args):
        super(Box2Comm, self).__init__()
        self.args = deepcopy(args)
        self.rpnpostprocess = RPNPostProcess(args['RPNPostProcess'])
    def get_cluster_idx(self, points_cluster_idx):
        max_idx = points_cluster_idx.max() + 1
        return torch.arange(0, max_idx).to(points_cluster_idx)
    def get_cluster_data(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_batchidx,
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
        cluster_batchidx = points_batchidx[cluster_idx][:]
        output_data['points'] = cluster_center[cluster_mask]
        output_data['points_proj'] = cluster_center_proj[cluster_mask]
        output_data['points_feature'] = cluster_feature[cluster_mask]
        output_data['logits'] = cluster_logits[cluster_mask]
        output_data['batch_idx'] = cluster_batchidx[cluster_mask]
        output_data['batch_size'] = cluster_batchidx[cluster_mask].max().item() + 1
        return output_data
    def forward(self, data_dict, output_dict):
        data_dict['box2comm'] = self.rpnpostprocess(data_dict['box2comm'], output_dict)
        fg_mask = (data_dict['box2comm']['point_box_indices']>=0)
        output_data_cluster = self.get_cluster_data(points=data_dict['box2comm']['points'],
                                                    points_proj=data_dict['box2comm']['points_proj'],
                                                    points_feature=data_dict['box2comm']['points_feature'],
                                                    points_logits=data_dict['box2comm']['points_logits'],
                                                    points_cluster_idx=data_dict['box2comm']['point_box_indices'],
                                                    points_batchidx=data_dict['box2comm']['box_batch_idx'],
                                                    fg_mask=fg_mask)
        return output_data_cluster
        
    
class Box2CommPoints(nn.Module):
    def __init__(self, args):
        super(Box2CommPoints, self).__init__()
        self.args = deepcopy(args)
        self.rpnpostprocess = RPNPostProcess(args['RPNPostProcess'])
    def get_cluster_idx(self, points_cluster_idx):
        max_idx = points_cluster_idx.max() + 1
        return torch.arange(0, max_idx).to(points_cluster_idx)
    def get_cluster_data(self, 
                         points, 
                         points_proj,
                         points_feature, 
                         points_logits,
                         points_cluster_idx, 
                         points_batchidx,
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
        cluster_batchidx = points_batchidx[cluster_idx][:]
        
        output_data['points'] = points_fore
        output_data['points_proj'] = points_proj_fore
        output_data['points_feature'] = cluster_feature[points_cluster_idx_fore]
        output_data['logits'] = cluster_logits[points_cluster_idx_fore]
        output_data['batch_idx'] = points_batchidx[fg_mask]
        output_data['batch_size'] = points_batchidx[fg_mask].max().item() + 1
        return output_data
    def forward(self, data_dict, output_dict):
        data_dict['box2comm'] = self.rpnpostprocess(data_dict['box2comm'], output_dict)
        fg_mask = (data_dict['box2comm']['point_box_indices']>=0)
        output_data_cluster = self.get_cluster_data(points=data_dict['box2comm']['points'],
                                                    points_proj=data_dict['box2comm']['points_proj'],
                                                    points_feature=data_dict['box2comm']['points_feature'],
                                                    points_logits=data_dict['box2comm']['points_logits'],
                                                    points_cluster_idx=data_dict['box2comm']['point_box_indices'],
                                                    points_batchidx=data_dict['box2comm']['box_batch_idx'],
                                                    fg_mask=fg_mask)
        return output_data_cluster