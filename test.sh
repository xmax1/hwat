#!/bin/sh

#SBATCH --cpus-per-task       4
#SBATCH --mem-per-cpu         1024
#SBATCH --error               dump/exp/demo-12/Wvyonoa/slurm/e-%j.err
#SBATCH --gres                gpu:RTX3090:2
#SBATCH --job-name            demo
#SBATCH --mail-type           FAIL
#SBATCH --nodes               1-1
#SBATCH --ntasks              2
#SBATCH --output              dump/exp/demo-12/Wvyonoa/slurm/o-%j.out
#SBATCH --partition           sm3090
#SBATCH --time                0-01:00:00

module purge
source ~/.bashrc
module load foss
module load CUDA/11.7.0
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
conda activate lumi
# 192GB 

srun --gres=gpu:RTX3090:1 --ntasks=1 --label --exact python run.py --run_name run.py --seed 808017424 --dtype float32 --n_step 2000 --log_metric_step 10 --log_state_step 10 --charge 0 --spin 0 --a [[0.0,0.0,0.0]] --a_z [4.0] --n_b 256 --n_corr 50 --n_equil 10000 --acc_target 0.5 --with_sign False --n_sv 32 --n_pv 16 --n_fb 2 --n_det 1 --terms_s_emb ['ra','ra_len'] --terms_p_emb ['rr','rr_len'] --ke_method vjp --job_type training --mail_type FAIL --partition sm3090 --nodes 1-1 --cpus_per_task 8 --time 0-01:00:00 --TMP dump/tmp --exp_id Wvyonoa --run_sweep False --user amawi --server svol.fysik.dtu.dk --git_remote origin --git_branch main --env lumi --debug False --wb_mode online --submit False --cap 3 --exp_path dump/exp/demo-12/Wvyonoa --exp_name demo --n_gpu 2 --device_type cuda --head True & 
srun --gres=gpu:RTX3090:1 --ntasks=1 --label --exact python run.py --run_name run.py --seed 808017424 --dtype float32 --n_step 2000 --log_metric_step 10 --log_state_step 10 --charge 0 --spin 0 --a [[0.0,0.0,0.0]] --a_z [4.0] --n_b 256 --n_corr 50 --n_equil 10000 --acc_target 0.5 --with_sign False --n_sv 32 --n_pv 16 --n_fb 2 --n_det 1 --terms_s_emb ['ra','ra_len'] --terms_p_emb ['rr','rr_len'] --ke_method vjp --job_type training --mail_type FAIL --partition sm3090 --nodes 1-1 --cpus_per_task 8 --time 0-01:00:00 --TMP dump/tmp --exp_id Wvyonoa --run_sweep False --user amawi --server svol.fysik.dtu.dk --git_remote origin --git_branch main --env lumi --debug False --wb_mode online --submit False --cap 3 --exp_path dump/exp/demo-12/Wvyonoa --exp_name demo --n_gpu 2 --device_type cuda --head False & 
wait 