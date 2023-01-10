#!/bin/bash
#SBATCH --job-name="rfm_job"
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition gpu
#SBATCH --account=<project>
#SBATCH --gpus-per-node=8

module load LUMI/22.08
module load partition/G
module load PyTorch

pip install paramiko jupyter ipykernel matplotlib

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

srun python cnn_distr.py