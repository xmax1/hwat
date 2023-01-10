from utils import Sub
from typing import Union
from pathlib import Path
import pprint
from utils import dict_to_cmd, cls_to_dict
from simple_slurm import Slurm
import os
from utils import run_cmds

# When using --cpus-per-task to run multithreaded tasks, be aware that CPU binding is inherited from the parent of the process. This means that the multithreaded task 
# should either specify or clear the CPU binding itself to avoid having all threads of the multithreaded task use the same mask/CPU as the parent. Alternatively, fat masks 
# (masks which specify more than one allowed CPU) could be used for the tasks in order to provide multiple CPUs for the multithreaded tasks.

# class CudaCMD:
#     pci_id: str = property(lambda _: ''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader')))

# class RocmCMD:
#     

CudaCMD = dict(
	pci_id: str = lambda _: ''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'))
)

RocmCMD = dict(
	pci_id: str = lambda _: ''.join(run_cmds('rocm-smi --query-gpu=pci.bus_id --format=csv,noheader'))
)

class nifl_slurm(Sub):
	export			= 'ALL'
	nodes           = '1' # (MIN-MAX) 
	mem_per_cpu     = 1024
	cpus_per_task   = 4
	gres            = property(lambda _: 'gpu:RTX3090:' + str(_._p.n_gpu))
	partition       = 'sm3090'
	ntasks          = property(lambda _: _._p.n_gpu)
	time            = '0-01:00:00'     # D-HH:MM:SS
	output          = property(lambda _: _._p.cluster_dir/'o-%j.out')
	error           = property(lambda _: _._p.cluster_dir/'e-%j.err')
	job_name        = property(lambda _: _._p.exp_name)
	# cpus_per_task   = 1
	# ntasks_per_node = 
	# tasks_per_gpu  = 8 
	# nodelist		= 's001,s005'
	# ntasks			= 40
	# mail_type       = 'FAIL'

	_job_id: 			str      = property(lambda _: os.environ['SLURM_JOBID'])
 
	def _sbatch(
		ii, 
		job: dict,
		run_name: Union[Path,str] = 'run.py',
		wandb_sweep: bool = False
	):
		mod = ['module purge', 'module load foss', 'module load CUDA/11.7.0']
		env = ['source ~/.bashrc', f'conda activate {ii._p.env}',]
		export = ['export $SLURM_JOB_ID',]
		debug = ['echo $SLURM_JOB_GPUS', 'echo $cluster_JOB_NODELIST', 'nvidia-smi']
		srun_cmd = 'srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label '
		sb = mod + env + debug

		n_dist = int(ii.nodes) * ii._p.n_gpu  # nodes much 
		for i in range(n_dist):
			device_log_path = ii._p.cluster_dir/(str(i)) # + ii._p.hostname.split('.')[0]+"_device.log")
			job['head'] = head = not bool(i)			
			cmd = dict_to_cmd(job)

			if wandb_sweep and head:
				cmd = f'wandb agent {ii._p.sweep_path_id} --count 1'
			else:
				cmd = f'python -u {run_name} {cmd}'
				
			sb += [f'{srun_cmd} {cmd} 1> {device_log_path} 2>&1 & ']
   
		sb += ['wait',]
		sb = '\n'.join(sb)
		return sb

	def _submit_to_cluster(ii, job):
		sbatch = ii._sbatch(job)
		d_c = cls_to_dict(ii, prop=True, ignore=ii._ignore)
		Slurm(**d_c).sbatch(sbatch)
  
class lumi_slurm(Sub):
	export			= 'ALL'
	nodes           = '1' # (MIN-MAX) 
	mem_per_cpu     = 1024
	cpus_per_task   = 4
	gres            = property(lambda _: 'gpu:RTX3090:' + str(_._p.n_gpu))
	partition       = 'eap'
	ntasks          = property(lambda _: _._p.n_gpu)
	time            = '0-00:10:00'     # D-HH:MM:SS
	output          = property(lambda _: _._p.cluster_dir/'o-%j.out')
	error           = property(lambda _: _._p.cluster_dir/'e-%j.err')
	job_name        = property(lambda _: _._p.exp_name)
	account			= 'project_465000153'
 
	#SBATCH --ntasks=4
	#SBATCH --ntasks-per-node=8
	#SBATCH --gpus-per-node=8
	#SBATCH --time=0:10:0
	#SBATCH --partition eap
	#SBATCH --account=project_465000153

	_job_id: 			str      = property(lambda _: os.environ['SLURM_JOBID'])
 
	def _sbatch(
		ii, 
		job: dict,
		run_name: Union[Path,str] = 'run.py',
		wandb_sweep: bool = False
	):

		mod = [
			'module purge', 
			'module load CrayEnv', 
			'module load PrgEnv-cray/8.3.3', 
			'module load craype-accel-amd-gfx90a',
			'module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules',
			'module load suse-repo-deps/sam-default'
			'module load rocm/sam-5.3.0.lua',
			'module load rccl/sam-develop.lua',
			'module load aws-ofi-rccl/sam-default.lua',
			'module load magma/sam-default.lua',
      	]
		env = ['source ~/.bashrc', f'conda activate {ii._p.env}',]
		export = [
			'export $SLURM_JOB_ID', 
			'export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3',
			'export NCCL_NET_GDR_LEVEL=3'
		]
		debug = ['echo $SLURM_JOB_GPUS', 'echo $SLURM_JOB_NODELIST', 'rocm-smi']
		
		srun_cmd = 'srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label '
		sb = mod + env + debug

		n_dist = int(ii.nodes) * ii._p.n_gpu  # nodes much 
		for i in range(n_dist):
			device_log_path = ii._p.cluster_dir/(str(i)) # + ii._p.hostname.split('.')[0]+"_device.log")
			job['head'] = head = not bool(i)			
			cmd = dict_to_cmd(job)

			if wandb_sweep and head:
				cmd = f'wandb agent {ii._p.sweep_path_id} --count 1'
			else:
				cmd = f'python -u {run_name} {cmd}'
				
			sb += [f'{srun_cmd} {cmd} 1> {device_log_path} 2>&1 & ']
   
		sb += ['wait',]
		sb = '\n'.join(sb)
		return sb

	def _submit_to_cluster(ii, job):
		sbatch = ii._sbatch(job)
		pprint.pprint(ii.__class__.__dict__)
		d_c = cls_to_dict(ii, prop=True, ignore=ii._ignore)
		Slurm(**d_c).sbatch(sbatch)
  
cluster_options = dict(
	nifl=nifl_slurm,
	lumi=lumi_slurm
)

engine_cmd_options = dict(
	cuda=CudaCMD,
	rocm=RocmCMD
)