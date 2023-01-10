#!/bin/bash
#SBATCH --job-name="rfm_job"
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition gpu
#SBATCH --account=<project>
#SBATCH --gpus-per-node=8

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load suse-repo-deps/sam-default
module load rocm/sam-5.3.3.lua
module load rccl/sam-develop-rocm-5.3.3.lua 
module load aws-ofi-rccl/sam-rocm-5.3.3.lua
module load magma/sam-rocm-5.3.3.lua

base conda env:
/pfs/lustrep2/projappl/project_462000125/samantao-public/apps-rocm-5.3.3/miniconda/conda.env.yml

wheels:
/pfs/lustrep2/projappl/project_462000125/samantao-public/apps-rocm-5.3.3/pytorch/torch-1.13.0a0+git5ef49fb-cp37-cp37m-linux_x86_64.whl
/pfs/lustrep2/projappl/project_462000125/samantao-public/apps-rocm-5.3.3/apex-vision/apex-0.1-cp37-cp37m-linux_x86_64.whl
/pfs/lustrep2/projappl/project_462000125/samantao-public/apps-rocm-5.3.3/apex-vision/torchvision-0.14.0a0+5ce4506-cp37-cp37m-linux_x86_64.whl


#!/bin/bash
#SBATCH --job-name=cnn-pytorch
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0:10:0
#SBATCH --partition eap
#SBATCH --account=project_465000153

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load suse-repo-deps/sam-default
module load rocm/sam-5.3.0.lua
module load rccl/sam-develop.lua
module load aws-ofi-rccl/sam-default.lua
module load magma/sam-default.lua

# export PATH="/users/adwilson/projects/environments/ts_demo/bin:$PATH”

#module load LUMI/22.08
#module load partition/G
#module load PyTorch
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

srun python cnn_distr.py