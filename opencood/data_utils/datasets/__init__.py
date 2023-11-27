# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.early_fusion_dataset_dair import EarlyFusionDatasetDAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_lidar_seg import IntermediateFusionDatasetDAIR_Lidar
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_lidar_seg_add_occlusion import IntermediateFusionDatasetDAIR_Lidar_occlusion
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_lidar_seg_add_occlusion_mask2former import IntermediateFusionDatasetDAIR_Lidar_occlusion_mask2former
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_lidar_seg_add_occlusion_b5former import IntermediateFusionDatasetDAIR_Lidar_occlusion_b5former
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_lidar_seg_add_occlusion_spvcnn import IntermediateFusionDatasetDAIR_Lidar_Occlusion_SPVCNN
from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR

__all__ = {
    'EarlyFusionDatasetDAIR': EarlyFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIR_Lidar': IntermediateFusionDatasetDAIR_Lidar,
    'IntermediateFusionDatasetDAIR_Lidar_occlusion':IntermediateFusionDatasetDAIR_Lidar_occlusion,
    'IntermediateFusionDatasetDAIR_Lidar_occlusion_Mask2former': IntermediateFusionDatasetDAIR_Lidar_occlusion_mask2former,
    'IntermediateFusionDatasetDAIR_Lidar_occlusion_B5formerr': IntermediateFusionDatasetDAIR_Lidar_occlusion_b5former,
    'IntermediateFusionDatasetDAIR_Lidar_Occlusion_SPVCNN': IntermediateFusionDatasetDAIR_Lidar_Occlusion_SPVCNN,
    'LateFusionDatasetDAIR': LateFusionDatasetDAIR
}

# the final range for evaluation
GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
