from utils import Sub
from typing import Union
from pathlib import Path
import pprint
from utils import dict_to_cmd, cls_to_dict, run_cmds
from simple_slurm import Slurm
import os

# When using --cpus-per-task to run multithreaded tasks, be aware that CPU binding is inherited from the parent of the process. This means that the multithreaded task 
# should either specify or clear the CPU binding itself to avoid having all threads of the multithreaded task use the same mask/CPU as the parent. Alternatively, fat masks 
# (masks which specify more than one allowed CPU) could be used for the tasks in order to provide multiple CPUs for the multithreaded tasks.

# class CudaCMD:
#     pci_id: str = property(lambda _: ''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader')))

# class RocmCMD:
#     

# sinfo -p cluster
# groups
# sbalance
# sreport -t hours cluster AccountUtilization account=project_465000153
# sbatch - submit a batch script
# salloc - allocate compute resources
# srun - allocate compute resources and launch job-steps
# squeue - check the status of running and/or pending jobs
# scancel - delete jobs from the queue
# sinfo - view intormation abount cluster nodes and partitions
# scontrol - show detailed information on active and/or recently completed jobs, nodes and partitions
# sacct - provide the accounting information on running and completed jobs
# slurmtop - text-based view of cluster nodes' free and in-use resources and status of jobs

# Based on available resources and in keeping with maintaining a fair balance between all users, we may sometimes be able to accommodate special needs for a limited time. In that case, please submit a short explanation to cluster-help@luis.uni-hannover.de.

# To list job limits relevant for you, use the sacctmgr command:

# sacctmgr -s show user
# sacctmgr -s show user adwilson (works on lumi)
# sacctmgr -s show user format=user,account,maxjobs,maxsubmit,maxwall,qos
# sacctmgr -s show user zailacka


# Up-to-date information on ALL available nodes:

#  sinfo -Nl
#  scontrol show nodes
# Information on partitons and their configuration:

#  sinfo -s
#  scontrol show partitions

CudaCMD = dict(
	pci_id=lambda _: ''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'))
)

RocmCMD = dict(
	pci_id=lambda _: ''.join(run_cmds('rocm-smi --query-gpu=pci.bus_id --format=csv,noheader'))
)

backend_cmd_all = dict(
	cuda=CudaCMD,
	rocm=RocmCMD
)

# class Cluster():
#	_ignore = ['submit', 'sbatch']

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
	
	def sbatch(
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
			device_log_path = ii._p.cluster_dir/(str(i)+"_device.log") # + ii._p.hostname.split('.')[0])
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

	def submit(ii, job):
		sbatch = ii.sbatch(job)
		d_c = cls_to_dict(ii, prop=True, ignore=ii._ignore)
		Slurm(**d_c).sbatch(sbatch)
  
  
#  standard-g: max 1024 GPU nodes, max runtime 48 hours. Exclusive mode, i.e., full nodes reserved.
#  small-g: max 4 GPU nodes, max runtime 72 hours. Possibility of allocating individual GPUs.
#  dev-g: max 16 GPU nodes, max runtime 6 hours.
class lumi_slurm(Sub):
	export			= 'ALL'
	nodes           = '1' # (MIN-MAX) 
	mem_per_cpu     = 1024
	cpus_per_task   = 4
	gres            = property(lambda _: 'gpu:RTX3090:' + str(_._p.n_gpu))
	partition       = 'dev-g'
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
		d_c = cls_to_dict(ii, prop=True, ignore=ii._ignore)
		Slurm(**d_c).sbatch(sbatch)
  
cluster_options = dict(
	nifl=nifl_slurm,
	lumi=lumi_slurm
)

