from utils import flat_any
import inspect
from typing import Callable
from functools import reduce, partial
from simple_slurm import Slurm
import wandb
from pathlib import Path
import sys
from pprint import pprint
from copy import copy
from typing import Any

from jax import numpy as jnp, random as rnd
from flax import linen as nn
import optax 

from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum, iterate_folder
from utils import flat_dict, mkdir, cmd_to_dict

from hwat import compute_s_perm, init_walker, compute_emb

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'

class af:
    tanh = nn.tanh

class Sub:
    _parent = None

    def __init__(_i, parent=None):
        _i._parent = parent
    
    @property
    def d(_i, ignore=['d', 'cmd', 'masks', '_parent', 'parent']):
        _d={} # becomes class variable in call line, accumulates
        for k,v in _i.__class__.__dict__.items():
            if k.startswith('_') or k in ignore:
                continue
            if isinstance(v, partial): 
                v = _i.__dict__[k]   # if getattr then calls the partial, which we don't want
            else:
                v = getattr(_i, k)
            _d[k] = copy(v)
        return _d

class Pyfig:

    project_root:   str     = Path().home()/'projects'
    project:        str     = 'hwat'

    data_dir:       Path    = project_root / 'data'  # not a property, might change
    run_path:       Path    = property(lambda _: _.project_path / 'run.py')
    exp_name:       str     = 'junk'
    exp_id:         str     = gen_alphanum(n=7)
    
    dtype:          str     = 'f32'
    n_step:         int     = 20 
    log_sample_step         = 5
    log_metric_step         = 5
    log_state_step          = 10          

    seed:           int     = 808017424 # grr
    rng_init:       jnp.ndarray = property(lambda _: rnd.split(rnd.PRNGKey(_.seed), _.n_device))
		
    class data(Sub):
        n_b:    int         = 16
        n_e:    int         = 4
        n_u:    int         = 2
        n_d:    int         = property(lambda _: _.n_e-_.n_u)
        a:      jnp.ndarray = jnp.array([[0.0, 0.0, 0.0],])
        a_z:    jnp.ndarray = jnp.array([4.,])
        init_walker = \
            property(lambda _: \
                partial(init_walker, n_b=_.n_b, n_u=_.n_u, n_d=_.n_d, center=_.a, std=0.1))
        corr_len:  int      = 20
        acc_target:int      = 0.5

    class model(Sub):
        n_sv: int       = 16
        n_pv: int       = 8
        n_fb: int       = 2
        n_det: int      = 1
        n_fb_out: int   = property(lambda _: _.n_sv*3+_.n_pv*2)

        terms_s_emb   = ['x_rlen']
        terms_p_emb   = ['xx']
        compute_s_emb = \
            property(lambda _: partial(compute_emb, terms=_.terms_s_emb))
        compute_p_emb = \
            property(lambda _: partial(compute_emb, terms=_.terms_p_emb))
        compute_s_perm: partial = \
            property(lambda _: partial(compute_s_perm, n_u=_._parent.data.n_u))
    
    class opt(Sub):
        optimizer       = 'Adam'
        b1              = 0.9
        b2              = 0.99
        eps             = 1e-8
        lr              = 0.001
        loss            = 'l1'  # change this to loss table load? 
        tx = property(lambda _: optax.adam(_._parent.opt.lr))

    class sweep(Sub):
        method          = 'random'
        name            = 'sweep'
        metrics         = dict(
            goal        = 'minimize',
            name        = 'validation_loss',
        )
        parameters = dict(
            batch_size  = {'values' : [16, 32, 64]},
            epoch       = {'values' : [5, 10, 15]},
            lr          = {'max'    : 0.1, 'min': 0.0001},
        )
        n_sweep         = reduce(
            lambda i,j:i*j,[len(v['values']) for k,v in parameters.items() if 'values' in v])+1
        sweep_id = ''

    class wandb(Sub):
        job_type        = 'training'
        entity          = 'xmax1'
        run_path        = None

    class slurm(Sub):
        mail_type       = 'FAIL'
        partition       ='sm3090'
        nodes           = 1                # n_node
        ntasks          = 8                # n_cpu
        cpus_per_task   = 1     
        time            = '0-12:00:00'     # D-HH:MM:SS
        gres            = 'gpu:RTX3090:1'
        output          = property(lambda _: _._parent.TMP /'o-%j.out')
        error           = property(lambda _: _._parent.TMP /'e-%j.err')
        job_name        = property(lambda _: _._parent.exp_name)  # this does not call the instance it is in
        sbatch          = property(lambda _: f""" 
            module purge 
            source ~/.bashrc 
            module load GCC 
            module load CUDA/11.4.1 
            module load cuDNN/8.2.2.26-CUDA-11.4.1 
            conda activate {_._parent.env} 
            export MKL_NUM_THREADS=1 
            export NUMEXPR_NUM_THREADS=1 
            export OMP_NUM_THREADS=1 
            export OPENBLAS_NUM_THREADS=1
            pwd
            nvidia-smi
            mv_cmd = f'mv {_._parent.TMP}/o-$SLURM_JOB_ID.out {_._parent.TMP}/e-$SLURM_JOB_ID.err $out_dir' 
    """
    )
    
    project_path:       Path    = property(lambda _: _.project_root / _.project)
    server_project_path:Path    = property(lambda _: _.project_path)
    n_device:           int     = property(lambda _: count_gpu())

    iter_exp_dir:       bool    = True
    project_exp_dir:    Path    = property(lambda _: _.project_path / 'exp')
    project_cfg_dir:    Path    = property(lambda _: _.project_path / 'cfg')
    exp_path:           Path    = property(lambda _: iterate_folder(_.project_exp_dir/_.exp_name,_.iter_exp_dir)/_.exp_id)
    TMP:                Path    = property(lambda _: _.project_exp_dir / 'tmp')

    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    user:               str     = 'amawi'     # SERVER
    entity:             str     = 'xmax1'       # WANDB entity
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        
    env:                str     = 'dex'            # CONDA ENV
    
    submit_state:       int  = -1
    _sys_arg:           list = sys.argv[1:]
    _wandb_ignore:      list = ['sbatch',]

    def __init__(_i,args:dict={},cap=40,wandb_mode='online',debug=False):
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                setattr(_i, k, v(_i))

        assert _i.merge(args|cmd_to_dict(sys.argv[1:],_i.d))

        mkdir(_i.exp_path)

        run = wandb.init(
            job_type    = _i.wandb.job_type,
            entity      = _i.wandb.entity,
            project     = _i.project,
            dir         = _i.exp_path,
            config      = _i._to_wandb(_i.d),
            mode        = wandb_mode,
            settings=wandb.Settings(start_method='fork'), # idk y this is issue, don't change
        )

        _i.wandb.run_path = run.path

        if _i.submit_state > 0:
            n_job_running = run_cmds([f'squeue -u {_i.user} -h -t pending,running -r | wc -l'])
            if n_job_running > cap:
                exit(f'There are {n_job_running} on the submit cap is {cap}')
            print(f'{n_job_running} on the cluster')

            _slurm = Slurm(**_i.slurm.d)
            n_run, _i.submit_state = _i.submit_state, 0            
            for _ in range(n_run):
                _slurm.sbatch(_i.slurm.sbatch \
                    + f'out_dir={(mkdir(_i.exp_path/"out"))} {_i.cmd} | tee $out_dir/py.out date "+%B %V %T.%3N" ')


    @property
    def cmd(_i,):
        d = flat_dict(_i.d)
        return ' '.join([f' --{k}  {str(v)} ' for k,v in d.items()])

    @property
    def commit_id(_i,)->str:
        process = run_cmds(['git log --pretty=format:%h -n 1'], cwd=_i.project_path)[0]
        return process.stdout.decode('utf-8') 

    def submit(_i, sweep=False, commit_msg=None, commit_id=None):
        commit_msg = commit_msg or _i.exp_id
        _i.submit_state *= -1
        if _i.submit_state > 0:
            if sweep:
                _i.sweep_id = wandb.sweep(
                    env     = f'conda activate {_i.env};',
                    sweep   = _i.sweep.d, 
                    program = _i.run_path,
                    project = _i.project,
                    name    = _i.exp_name,
                    run_cap = _i.sweep.n_sweep
                )
                _i.submit_state *= _i.sweep.n_sweep
            
            # local_out = run_cmds(['git add .', f'git commit -m {commit_msg}', 'git push'], cwd=_i.project_path)
            cmd = f'python -u {_i.run_path} ' + (commit_id or _i.commit_id) + _i.cmd
            # server_out = run_cmds_server(_i.server, _i.user, cmd, cwd=_i.server_project_path)
    
    @property
    def d(_i, _ignore_attr=['d', 'cmd', 'submit', 'partial']):
        _d = {}
        for k,v in _i.__class__.__dict__.items():
            if k.startswith('_') or k in _ignore_attr:
                continue
            v = getattr(_i, k)
            if isinstance(v, Sub):
                v = v.d
            _d[k] = v
        return _d

    @property
    def _sub_cls(_i):
        return [v for v in _i.__dict__.values() if isinstance(v, Sub)]

    def partial(_i, f:Callable):
        d = flat_any(_i.d)
        d_k = inspect.signature(f.__init__).parameters.keys()
        d = {k:v for k,v in d.items() if k in d_k}
        return f(**d)

    def merge(_i, d:dict, _n=0):
        for k,v in d.items():
            for cls in [_i]+_i._sub_cls:
                if k in cls.__dict__:
                    cls.__dict__[k] = copy(v)
                    _n += 1
        return (_n - len(d))==0

    def _to_wandb(_i, d:dict, parent='', _l:list=[])->dict:
        sep='.'
        for k, v in d.items():
            if isinstance(v, Path) or callable(v):
                continue

            if k in _i._wandb_ignore:
                continue
            
            k_1 = parent + sep + k if parent else k
            
            if isinstance(v, dict):
                _l.extend(_i._to_wandb(v, k_1, _l=_l).items())
            elif callable(v):
                continue
            _l.append((k_1, v))
        return dict(_l)


