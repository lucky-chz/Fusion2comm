#!/bin/bash
#SBATCH -o /experiment0/chuhz/project/outputs/where2comm_logs/b5former/logs_temp/dair_where2comm_max_multiscale_resnet_logits_fuse3d_da_kl_0.2_add_occlusion_1_%j.out
#SBATCH -p 3090
#SBATCH --exclude=3dimage-13
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=80G

srun python3 opencood/tools/inference.py --model_dir /experiment0/chuhz/project/outputs/where2comm_logs/logs/dair_where2comm_max_multiscale_resnet_baseline_2023_06_06_21_20_20 --fusion_method intermediate_with_comm --comm_thre 0.001
# srun python opencood/tools/train.py --hypes_yaml /space/chuhz/workspace/v2x_object/Where2comm/opencood/hypes_yaml/dair-v2x/segformerb5/dair_where2comm_max_multiscale_resnet_logits_fuse3d_da_ce_0.2_add_occlusion_1.yaml

# CUDA_VISIBLE_DEVICES=1 python3 opencood/tools/inference.py --model_dir /experiment0/chuhz/project/outputs/where2comm_logs/logs/dair_where2comm_max_multiscale_resnet_baseline_2023_06_06_21_20_20 --fusion_method intermediate_with_comm --comm_thre 0.0001