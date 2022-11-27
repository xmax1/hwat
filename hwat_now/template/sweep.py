import wandb
from walle.pyfig import Pyfig
from walle.bureaucrat import remove_path
# run_cmd(f'export WANDB_DIR={c.exp_path}', cwd='.')  # set wandb before the path changes

c = Pyfig(remote=True, sweep=True, sweep_params=None)

"""
TODO
make gpus available if local

NOTES
For runs that are not part of a sweep, the values of wandb.config are usually set by providing a dictionary 
to the config argument of wandb.init. During a sweep, however, any configuration information passed to wandb.init 
is instead treated as a default value, which might be over-ridden by the sweep.

command:
  - ${env}   # activates /usr/bin/env
  - ${interpreter}
  - ${program}
  - ${args}

/usr/bin/env python ${program} --param1=value1 --param2=value2
"""
