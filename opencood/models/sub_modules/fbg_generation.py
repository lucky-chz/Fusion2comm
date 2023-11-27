import torch
import torch.nn as nn
import numpy as np

class ForegroundMaskGenerator(nn.Module):
    def __init__(self, args):
        super(ForegroundMaskGenerator, self).__init__()
        self.args = args
        self.cls_list = args['cls_list']
        self.thresh_list = args['thresh_list']
    def forward(self, data_dict):
        logits = data_dict['logits']
        score = torch.softmax(logits, dim=1)
        thresh = torch.from_numpy(np.array(self.thresh_list)).float().reshape(1, -1).to(score)
        score_select = score[:, self.cls_list]
        score_mask = score_select>thresh
        score_cnt = score_mask.sum(dim=1)
        fg_mask = score_cnt>0
        if self.training:
            fg_mask = score_cnt>=0
            fg_mask = (data_dict['instance_label']>=0) | fg_mask
        # else: # 
            # fg_mask = (data_dict['instance_label']>=0)
        
        data_dict['foreground_mask'] = fg_mask
        return data_dict