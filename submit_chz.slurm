#!/bin/bash
#SBATCH -o /experiment0/chuhz/project/outputs/where2comm_logs/b5former/logs/dair_where2comm_max_multiscale_resnet_logits_fuse3d_da_ce_0.2_add_occlusion_1_%j.out
#SBATCH -p a6000
#SBATCH --exclude=3dimage-20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=80G

srun python opencood/tools/train.py --hypes_yaml /space/chuhz/workspace/v2x_object/Where2comm/opencood/hypes_yaml/dair-v2x/segformerb5/dair_where2comm_max_multiscale_resnet_logits_fuse3d_da_ce_0.2_add_occlusion_1.yaml