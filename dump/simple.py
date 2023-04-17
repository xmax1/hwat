from things.core import mkdir, this_is_noop
from things.utils import npify_tree, compute_metrix

import wandb

import numpy as np

from things.utils import Metrix
from things.core import lo_ve

from pyfig import Pyfig

# from dashboard import run_dash
# from pyfig import Pyfig

from run import init_exp, update_params, update_grads

c_init = dict(
	exp_name = 'simple',
	n_e = 5
) # config variables dictionary

yaml_config_path = 'config.yaml' # other config variables
c_yaml = lo_ve(yaml_config_path)

c_init.update(c_yaml)

c = Pyfig(notebook=False, sweep= None, c_init= c_init)  # config from command line, file, dictionary, and defaults


# dashboard = run_dash(c.d) # and dashboard
# dashboard.run_server(debug= False)
# exit()
# c.update(c_yaml) # update at any time



model, dataloader, compute_loss, opt, scheduler = init_exp(c)  # all the things
model.requires_grad_(True)

init_summary = dict()
metrix = Metrix(c.mode, init_summary, c.opt_obj_key, opt_obj_op= c.opt_obj_op)  # for collecting metrics and logging

c.start()  # start timer, wandb 



v_cpu_d = dict()
for step, loader_d in enumerate(dataloader, start= 1):  # dataloader contains sampling loop

	model.zero_grad(set_to_none= True)  # useful in general, here it is essential bc loss function weird

	# LOSS
	loss, v_d = compute_loss(step, **loader_d, model= model, debug= c.debug, c= c)  # general. v_d contains everything

	# DISTRIBUTION
	v_d = c.dist.sync(
		v_d, 
		sync_method= c.mean_tag, 
		this_is_noop= this_is_noop(step, c.n_step, every= c.dist.sync_every)
	)  # where the distribution happens if needed

	# UPDATE
	v_d = update_grads(step, model, v_d, debug= c.debug, c= c) 
	update_params(model, opt, scheduler)

	# LOGGING
	v_d.update( (('params', {k:p.detach() for k,p in model.named_parameters()}),))  # collecting for logging

	v_cpu_d: dict = npify_tree(v_d)
	
	v_metrix = metrix.tick(step, v_cpu_d= v_cpu_d)
	v_metrix = compute_metrix(v_metrix, sep= '/', debug= c.debug)  # reformatting for wandb
	wandb.log(v_metrix, step= step, commit= True)  # tada! 



c.end()  # fin

        
			

