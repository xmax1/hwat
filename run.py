import wandb
from pyfig import Pyfig

c = Pyfig(remote=False, sweep=False)

run = wandb.init(  # not needed for sweep
    entity=c.entity,
    project=c.project,
    job_type=c.job_type,
    cfg=c.dict,  # over-ridden in sweep case
)

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
