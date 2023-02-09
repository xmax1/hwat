from pathlib import Path
from .utils import PlugIn

class PathsBase(PlugIn):
	home = Path()
	@property
	def project_dir(ii) -> Path: 
		return ii.home / 'projects' / ii.p.project
	@property
	def dump_dir(ii) -> Path:
		return Path('dump')
	@property
	def tmp_dir(ii) -> Path:
		return Path(ii.dump_dir,'tmp')
	@property
	def exp_dir(ii) -> Path:
		return Path(ii.dump_exp_dir, ii.p.exp_name, ii.p.exp_id)
	@property
	def dump_exp_dir(ii) -> Path:
		return Path(ii.dump_dir, 'exp')
	@property
	def cluster_dir(ii) -> Path:
		return Path(ii.exp_dir, 'cluster')
	@property
	def exchange_dir(ii) -> Path:
		return Path(ii.exp_dir, 'exchange')
	@property
	def state_dir(ii) -> Path:
		return Path(ii.exp_dir, 'state')
	@property
	def code_dir(ii) -> Path:
		return Path(ii.exp_dir, 'code')
	@property
	def exp_data_dir(ii) -> Path:
		return Path(ii.exp_dir, 'exp_data')
		
class DataBase(PlugIn):
	n_b:        int         = 1
	loader_n_b: int 		= 1
	
class OptBase(PlugIn):
	opt_name: 		str		= ''
	lr:  			float 	= 1.
	betas:			list	= (0.9, 0.99)
	eps: 			float 	= 1.
	weight_decay: 	float 	= 1.
	hessian_power: 	float 	= 1.

class SchedulerBase(PlugIn):
	scheduler_name: str = ''
	step_size: int = 1
	gamma: float = 1.

class ModelBase(PlugIn):
	pass