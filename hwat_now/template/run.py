import wandb
import numpy as np

# If from some weird location
# from walle.bureaucrat import load_pyfig
# c = load_pyfig(cfg_path='./pyfig.py', iterate=False) # This is the base config
# Otherwise
from pyfig import Pyfig

c = Pyfig(remote=False, sweep=False)

run = wandb.init(  # not needed for sweep
    entity=c.entity,
    project=c.project,
    job_type=c.job_type,
    cfg=c.dict,  # over-ridden in sweep case
)

def log_metric(
    *, 
    model  = None, 
    param  = None, 
    metric = []
):
    metric_fn = {
        'mse': lambda x: x,
        'l1_norm': lambda x, y: np.mean(np.abs(x - y))
    }
    for m in metric:
        metric_fn[m](param)

metric = ['loss', 'this_thing']

for step, batch in range(1, c.n_step+1):
    
    if c.log_metric_step % step == 0:
        metric = log_metric(model=model, param=param, metric=metric)
        wandb.log({
                'train/step': step, 
                **summary
        })
    
    if c.log_state_step % step == 0:
        artifact = wandb.Artifact(name=f'model-{wandb.run.id}', type='')
        artifact.add_file(c.exp_path / f'model_i{step}')
        wandb.run.log_artifact(artifact)
