

from pathlib import Path
from .utils import PlugIn
from .core import TryImportThis

class LoggerBase(PlugIn):
	
	run = None
	
	job_type:		str		= ''		
	log_mode: 		str		= ''
	log_run_path:	str 	= ''
	log_run_type: 	str		= property(lambda _: f'groups/{_.p.exp_name}/workspace') 
	log_run_url: 		str		= property(lambda _: f'{_.p.project}/{_.log_run_type}')

	def get_c(ii, d:dict)->dict:
		return d

	def start(ii, d: dict, run_id: str, tags: list):
		return

	def end(ii):
		return
		

def d_to_wb(d:dict, parent='', sep='.', items:list=None)->dict:
	items = items or []
	for k, v in d.items():
		if callable(v):
			continue
		elif isinstance(v, Path):
			parent = 'path' + sep
		elif isinstance(v, dict):
			items.extend(d_to_wb(v, parent=k, items=items).items())
		name = parent + k
		items.append((name, v))
	return dict(items)

Wandb = None
with TryImportThis('wandb') as _wb:
	
	import wandb

	class Wandb(PlugIn):
		
		run = None
		entity:			str		= property(lambda _: _.p.entity)
		program: 		Path	= property(lambda _: Path( _.p.paths.project_dir, _.p.run_name))
		
		job_type:		str		= ''		
		log_mode: 		str		= ''
		log_run_path:	str 	= ''
		log_run_type: 	str		= property(lambda _: f'groups/{_.p.exp_name}/workspace') 
		log_run_url: 		str		= property(lambda _: f'https://wandb.ai/{_.p.entity}/{_.p.project}/{_.log_run_type}')


		def get_c(ii, d:dict, parent='', sep='.', items:list=None)->dict:
			return d_to_wb(d, parent, sep, items)

		def start(ii, d: dict, run_id: str, tags: list):
			print('pyfig:start: logger_run_path = \n ***', ii.log_run_path, '***')
			ii.log_run_path = f'{ii.entity}/{ii.p.project}/{run_id}'  # no no :,/ - in mode _ in names try \ + | .
			
			ii.run = wandb.init(
				project     = ii.p.project, 
				group		= ii.p.exp_name,
				dir         = ii.p.paths.exp_data_dir,
				entity      = ii.entity,	
				mode        = ii.log_mode,
				config      = ii.get_c(d, parent='', sep='.'),
				id			= run_id,
				tags 		= tags,
				reinit 		= not (ii.run is None),
			)

		def end(ii):
			try:
				ii.run.finish()
			except Exception as e:
				print('logger:end:error: (iz ok) ', e)
				


			
			
		
		
		
		
		
		
		
		