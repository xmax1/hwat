


from pathlib import Path
import os
from simple_slurm import Slurm
import math 

from .core_utils import run_cmds, dict_to_cmd
from .utils import PlugIn


this_dir = Path(__file__).parent
hostname = os.environ.get('HOSTNAME', None)

class ResourceBase(PlugIn):
	n_gpu: int = 0
	n_node: int = 1
	n_thread_per_process: int = 1

	def cluster_submit(ii, job: dict):
		return job
	
	def device_log_path(ii, rank=0):
		if not rank:
			return ii.p.paths.exp_dir/(str(rank)+"_device.log") 
		else:
			return ii.p.paths.cluster_dir/(str(rank)+"_device.log")

class Niflheim(ResourceBase):

	n_gpu: int 		= 1
	n_node: int 	= property(lambda _: int(math.ceil(_.n_gpu/10))) 
	n_thread_per_process: int = property(lambda _: _.slurm_c.cpus_per_gpu)

	architecture:   	str 	= 'cuda'
	nifl_gpu_per_node: 	int  = property(lambda _: 10)

	job_id: 		str  	= property(lambda _: os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.'))  # slurm only

	_pci_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
	pci_id:			str		= property(lambda _: ''.join(run_cmds(_._pci_id_cmd, silent=True)))

	n_device_env:	str		= 'CUDA_VISIBLE_DEVICES'
	# n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, '')))
	n_device:       int     = property(lambda _: len(os.environ.get(_.n_device_env, '').replace(',', '')))


	class slurm_c(PlugIn):
		export			= 'ALL'
		cpus_per_gpu    = 8				# 1 task 1 gpu 8 cpus per task 
		partition       = 'sm3090'
		time            = '0-00:10:00'  # D-HH:MM:SS
		nodes           = property(lambda _: str(_.p.n_node)) 			# (MIN-MAX) 
		gres            = property(lambda _: 'gpu:RTX3090:' + str(min(10, _.p.n_gpu)))
		ntasks          = property(lambda _: _.p.n_gpu if int(_.nodes)==1 else int(_.nodes)*80)
		job_name        = property(lambda _: _.p.p.exp_name)
		output          = property(lambda _: _.p.p.paths.cluster_dir/'o-%j.out')
		error           = property(lambda _: _.p.p.paths.cluster_dir/'e-%j.err')

	# mem_per_cpu     = 1024
	# mem				= 'MaxMemPerNode'
	# n_running_cmd:	str		= 'squeue -u amawi -t pending,running -h -r'
	# n_running:		int		= property(lambda _: len(run_cmds(_.n_running_cmd, silent=True).split('\n')))	
	# running_max: 	int     = 20

	def cluster_submit(ii, job: dict):

		# module load foss
		body = []
		body = f"""
		module purge
		source ~/.bashrc
  		conda activate {ii.p.env}
		export exp_id="{job["exp_id"]}"
		echo exp_id-${{exp_id}}
		MKL_THREADING_LAYER=GNU   
		export ${{MKL_THREADING_LAYER}}
		"""
		# important, here, hf_accel and numpy issues https://github.com/pytorch/pytorch/issues/37377

		extra = """
		module load CUDA/11.7.0
		module load OpenMPI
		export MKL_NUM_THREADS=1
		export NUMEXPR_NUM_THREADS=1
		export OMP_NUM_THREADS=8
		export OPENBLAS_NUM_THREADS=1
		export TORCH_DISTRIBUTED_DEBUG=DETAIL

		echo $PWD
		"""
		
		debug_body = f""" \
  		export $SLURM_JOB_ID
		echo all_gpus-${{SLURM_JOB_GPUS}}', 'echo nodelist-${{SLURM_JOB_NODELIST}}', 'nvidia-smi']
		echo api-${{WANDB_API_KEY}}
		echo project-${{WANDB_PROJECT}}
		echo entity-${{WANDB_ENTITY}}
		echo ${{PWD}}
		echo ${{CWD}}
		echo ${{SLURM_EXPORT_ENV}}
		scontrol show config
		srun --mpi=list
		export WANDB_DIR="{ii.p.paths.exp_dir}"
		printenv']
		curl -s --head --request GET https://wandb.ai/site
		ping api.wandb.ai
		"""
		# CUDA_LAUNCH_BLOCKING=1 

		# body += debug_body
		# import pprint
		# pprint.pprint(flat_any(job))

		body += extra

		nodes = iter(range(ii.n_node))
		for submit_i in range(ii.p.dist.n_launch):
			if submit_i % 10 == 0:
				node_i = next(nodes)

			print(f'\ndistribution: \t {ii.p.dist.dist_name}',  f'submit_i: \t\t {submit_i}', f'node_i: \t\t {node_i}')

			cmd = dict_to_cmd(job, exclude_none=True)

			body += f'{ii.p.dist.launch_cmd(node_i, submit_i, cmd)} 1> {ii.device_log_path(rank=submit_i)} 2>&1 & \n'
			
		body += '\nwait \n'
		body += '\necho End \n'

		# body += [f'wandb accel {ii.p.wb.sweep_id}'] # {ii.p.wb.sweep_path_id}
		# body += [f'wandb agent {ii.p.wb.sweep_id} 1> {ii.device_log_path(rank=0)} 2>&1 ']
		# body += ['wait',]
	
		body = body.split('\n')
		body = [b.strip() for b in body]
		body = '\n'.join(body)
		
		ii.p.log([body,], ii.p.paths.cluster_dir/'sbatch.log')
		job_id = ii._slurm.sbatch(body, verbose=True)
		print('slurm out: ', job_id)
  
	@property
	def _slurm(ii,) -> Slurm:
		ii.p.if_debug_print_d(ii.slurm_c.d)
		return Slurm(**ii.slurm_c.d)
	
	
	
	

