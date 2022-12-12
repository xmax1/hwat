

sbatch_cmd = f""" 
module purge 
source ~/.bashrc 
module load GCC 
module load CUDA/11.4.1 
module load cuDNN/8.2.2.26-CUDA-11.4.1 
conda activate {_._p.env} 
export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OMP_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
pwd
nvidia-smi
mv_cmd = f'mv {_._p.TMP}/o-$SLURM_JOB_ID.out {_._p.TMP}/e-$SLURM_JOB_ID.err $out_dir' 
"""