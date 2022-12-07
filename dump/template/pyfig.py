from pathlib import Path
from walle.bureaucrat import iterate_folder, gen_alphanum, mkdir

import sys
from pathlib import Path
from walle.bureaucrat import mkdir, gen_alphanum
from walle.submit import count_gpu, git_commit_id
from simple_slurm import Slurm
from functools import reduce
import wandb

TMP = mkdir('./tmp/out')

sweep = dict(
    method = 'random',
    name = 'sweep',
    metrics = dict(
        goal = 'minimize',
        name = 'validation_loss',
    ),
    parameters = {}
    # parameters = dict(
    #     batch_size = {'values': [16, 32, 64]},
    #     epoch = {'values': [5, 10, 15]},
    #     lr = {'max': 0.1, 'min': 0.0001},
    # ),
)

class Pyfig:

    sys_arg: dict = sys.argv[1:]

    seed:               int     = 808017424         # grr
    env:                str     = ''                # CONDA ENV
    commit_id:          str     = git_commit_id()
    n_device:           int     = property(lambda _: count_gpu())
    job_type:           str     = 'training'

    exp_name:           str     = exp_name
    run_path:           Path    = property(lambda _: _.project_dir / 'run.py')
    exp_id:             str     = gen_alphanum(n=7)
    
    sweep_id:           str     = ''
    sweep = sweep
    n_sweep: int = reduce(lambda i0,i1: i0*i1, [len(v) for v in sweep['parameters'].values()]) if sweep['parameters'] else 0

    dtype:              str     = 'f32'
    n_step:             int     = 100
    b_size:             int     = 32
    lr:                 float   = 0.001

    n_layer:            int     = 3
    af:                 str     = 'tanh'    # activation function

    log_metric_step:    int     = 100
    log_state_step:     int     = 1000         # wandb entity

    project:            str     = ''
    project_dir:        Path    = Path().absolute().parent
    server_project_dir: Path    = property(lambda _: _.project_dir)

    project_exp_dir:    Path    = property(lambda _: _.project_dir / 'exp')
    project_cfg_dir:    Path    = property(lambda _: _.project_dir / 'cfg')
    iter_exp_dir:       bool    = True
    exp_path:           Path    = property(lambda _: iterate_folder(_.project_exp_dir, _.exp_name, _.iter_exp_dir) / _.exp_id)

    data_path:          Path    = Path('~/data/a_data_file')
    server:             str     = 'server_id'   # SERVER
    user:               str     = 'user_id'     # SERVER
    entity:             str     = ''            # WANDB entity
    git_remote:         str     = 'origin'
    git_branch:         str     = 'main'

    slurm: Slurm = Slurm()
    sbatch: property = sbatch

    def __init__(self):
        mkdir(self.exp_path)

    def run_remote(self, sweep=False):
        if sweep:
            if self.sweep_id:
                cmd = self.cfg_to_cmd()
                self.run_slurm(cmd=cmd)
                exit('Run submitted to slurm') 
            else:  
                
                [self.run_slurm() for _ in range(self.n_sweep)]
                exit('Sweep submitted to slurm')         

    def run_sweep_agent(self, cmd: dict = {}, sweep=False):
        if self.sweep_id:
            return wandb.agent(self.sweep_id, count=1)

        if not sweep:
            self.n_sweep = 1
            cmd = self.dict_to_cmd(cmd)
        
        else:
            self.sweep_id = f' --sweep_id ' + wandb.sweep(
                env     = f'conda activate {self.env};',
                sweep   = self.sweep, 
                program = self.run_path,
                project = self.project,
                name    = self.exp_name,
                run_cap = self.n_sweep
            ) 

        cmd = f'python -u {self.run_path} ' + self.sweep_id + cmd

        for _ in range(self.n_sweep):
            self.slurm.sbatch(
                self.sbatch + 
                f'out_dir={(mkdir(self.exp_path/"out"))} \n \
                {cmd} | tee $out_dir/py.out \n \
                date "+%B %V %T.%3N" '
            )

        exit(f'{self.n_sweep} job submitted to slurm, well done')
  
    @property
    def dict(self,):
        d = self._dict_from_cls(self)
        for k,v in d.items():
            if isinstance(v, Slurm):
                d[k] = self._dict_from_cls(v)
        return d
        
    @staticmethod
    def _dict_from_cls(cls, get_prop): # GETS ALL ATTRS AND PROPERTIES 
        if not get_prop:
            return cls.__dict__
        ps = {k: getattr(cls, k) for k,v in super.__dict__.items() 
                    if isinstance(v, property) and not k=='dict'}
        return cls.__dict__ | ps

    def update_from_sys_arg(self):
        booleans = [arg for i, arg in self.sys_arg if self.sys_arg[i].startswith('-') and self.sys_arg[i+1].startswith('-')]
        d = {b:True for b in booleans}
        sys_arg = iter([arg for arg in self.sys_arg if not arg in booleans])
        
        for k,v in sys_arg:
            d[k] = v
        self.update(d)

    def update(self, d: dict):
        for k,v in d.items():
            assert self._update(self, self, k, v)

    def _update(self, cls, k, v):
        if k in cls.__dict__.keys():
            cls.__dict__[k] = type(cls.__dict__[k])(v)
            return True
        for cls_v in cls.__dict__.values():
            if isinstance(cls_v, Slurm):
                self._update(cls_v, k, v)
        return False

    def cfg_to_cmd(self):
        d = self._dict_from_cls(self, get_prop=False)
        return self.dict_to_cmd(d)

    @staticmethod
    def dict_to_cmd(d:dict):
        return {f'--{k}':str(v) for k,v in d.items()}

slurm = Slurm(
    output          = TMP / 'o-%j.out',
    error           = TMP / 'e-%j.err',
    mail_type       = 'FAIL',
    partition       ='sm3090',
    nodes           = 1,                # n_node
    ntasks          = 8,                # n_cpu
    cpus_per_task   = 1,     
    time            = '0-12:00:00',     # D-HH:MM:SS
    gres            = 'gpu:RTX3090:1',
    job_name        = property(lambda _: _.exp_name),  # this does not call the instance it is in
)

sbatch = property(lambda _: 
f""" 
module purge 
source ~/.bashrc 
module load GCC 
module load CUDA/11.4.1 
module load cuDNN/8.2.2.26-CUDA-11.4.1 
conda activate {_.env} 
export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OMP_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
pwd
nvidia-smi
mv_cmd = f'mv {TMP}/o-$SLURM_JOB_ID.out {TMP}/e-$SLURM_JOB_ID.err $out_dir' 
"""
)
