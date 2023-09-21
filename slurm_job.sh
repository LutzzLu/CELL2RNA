#!/bin/bash
#SBATCH --chdir=/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/all_code
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=100:00:00
#SBATCH --job-name=TCP
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --account=qdp-alpha
#SBATCH --partition=v100_12
#SBATCH --exclude=gv01
#SBATCH --gpu_cmode=shared
source ~/.bashrc
cd /dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/all_code

conda activate jupyter
echo GPU=$CUDA_VISIBLE_DEVICES
hostname
python3 /dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/all_code/0_subgraph.py
    