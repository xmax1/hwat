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
import shutil

from jax import numpy as jnp
from jax import random as rnd
import numpy as np
from flax import linen as nn
import optax 

from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum, iterate_folder
from utils import flat_dict, mkdir, cmd_to_dict

from hwat import compute_s_perm, init_walker, compute_emb, create_masks

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'
success = ["❌", "✅"]

# Variables cannot be a type
# Until we understand the jit leak compilation issue - we only use np arrays. 



class af:
    tanh = nn.tanh

class Sub:
    _parent = None

    def __init__(_i, parent=None):
        _i._parent = parent
    
    @property
    def d(_i, ignore=['d', 'cmd', '_parent']):
        _d={} # becomes class variable in call line, accumulates
        for k,v in _i.__class__.__dict__.items():
            if k.startswith('_') or k in ignore:
                continue
            if isinstance(v, partial): 
                v = _i.__dict__[k]   # if getattr then calls the partial, which we don't want
            else:
                v = getattr(_i, k)
            _d[k] = v
        return _d



class Pyfig:
    # 🔴 Variables in any class/subclass cannot have the same name

    project_root:   str     = Path().home()/'projects'
    project:        str     = 'hwat'

    data_dir:       Path    = project_root / 'data'  # not a property, might change
    run_path:       Path    = property(lambda _: _.project_path / 'run.py')
    exp_name:       str     = 'demo-final'
    exp_id:         str     = gen_alphanum(n=7)
    
    dtype:          str     = 'float32'
    n_step:         int     = 10000
    log_metric_step:int     = 100
    log_state_step: int     = 10          

    # seed:           int     = 808017424 # grr
    seed:           int     = 80085 # grr
    # rng_init:   jnp.ndarray = property(lambda _: rnd.split(rnd.PRNGKey(_.seed), _.n_device))
		
    class data(Sub):
        with_sign: bool     = False
        n_b:    int         = 512

        l_e:    list        = [4,]
        n_u:    int         = 2

        n_e:    int         = property(lambda _: int(sum(_.l_e)))
        n_d:    int         = property(lambda _: _.n_e-_.n_u)
        a:      np.ndarray  = property(lambda _: np.array([[0.0, 0.0, 0.0],]))
        a_z:    np.ndarray  = property(lambda _: np.array([4.,]))
        
        init_walker = \
            property(lambda _: \
                partial(init_walker, n_b=_.n_b, n_u=_.n_u, n_d=_.n_d, center=_.a, std=0.1))
        corr_len:   int      = 20
        equil_len:  int      = 10000  # total number: n_equil_loop = equil_len / corr_len
        acc_target: int      = 0.5
        
    class model(Sub):
        n_sv: int       = 32
        n_pv: int       = 16
        n_fbv: int      = property(lambda _: _.n_sv*3+_.n_pv*2)
        n_fb: int       = 3
        n_det: int      = 1
        terms_s_emb:list     = ['r', 'ra',]
        terms_p_emb:list     = ['rr',]
        compute_s_emb:Callable   = property(lambda _: partial(compute_emb, terms=_.terms_s_emb, a=_._parent.data.a))
        compute_p_emb:Callable   = property(lambda _: partial(compute_emb, terms=_.terms_p_emb))
        # p_mask_u: jnp.ndarray    = property(lambda _: create_masks(_._parent.data.n_e, _._parent.data.n_u, _._parent.dtype))
        # p_mask_d: jnp.ndarray    = property(lambda _: create_masks(_._parent.data.n_e, _._parent.data.n_u, _._parent.dtype))
        # compute_s_perm: Callable = property(lambda _: partial(compute_s_perm, n_u=_._parent.data.n_u, p_mask_u=_.p_mask_u, p_mask_d=_.p_mask_d))
    
    class opt(Sub):
        optimizer       = 'Adam'
        b1              = 0.9
        b2              = 0.99
        eps             = 1e-8
        lr              = 0.0001
        loss            = 'l1'  # change this to loss table load? 
        tx = property(lambda _: optax.chain(optax.adaptive_grad_clip(0.1), optax.adam(_.lr)))

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

    class wandb_c(Sub):
        job_type        = 'training'
        entity          = 'xmax1'
        wandb_run_path  = ''

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

    iter_exp_dir:       bool    = False  # True is broken, bc properties
    exp_path:           Path    = property(lambda _: iterate_folder(_.project_exp_dir/_.exp_name,_.iter_exp_dir)/_.exp_id)
    project_exp_dir:    Path    = property(lambda _: _.project_path / 'exp')
    project_cfg_dir:    Path    = property(lambda _: _.project_path / 'cfg')
    
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

    def __init__(_i,args:dict={},cap=40,wandb_mode='online',get_sys_arg=True):
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=_i)
                setattr(_i, k, v)
        
        _i.merge(cmd_to_dict(sys.argv[1:], _i.d))
        _i.merge(args)

        mkdir(_i.exp_path)
        print('Path: ', _i.exp_path.absolute(), '✅')
        
        print('System ')
        pprint(_i.data.d)
        print('Model ')
        pprint({k:(v if not k=='masks' else '...') for k,v in _i.model.d.items() })

        run = wandb.init(
            job_type    = _i.wandb_c.job_type,
            entity      = _i.wandb_c.entity,
            project     = _i.project,
            dir         = _i.exp_path,
            config      = _i._to_wandb(_i.d),
            mode        = wandb_mode,
            settings=wandb.Settings(start_method='fork'), # idk y this is issue, don't change
        )

        _i.wandb_c.wandb_run_path = run.path  
        print('run: ', run.path, '✅')

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
    def d(_i, _ignore_attr=['d', 'cmd', 'submit', 'partial', 'test_suite', 'sweep']):
        _d = {}
        for k,v in _i.__class__.__dict__.items():
            if k.startswith('_') or k in _ignore_attr:
                continue
            if isinstance(v, partial):
                v = copy(_i.__dict__[k]) # ‼ 🏳 Danger zone - partials may not be part of dict
            else:
                v = getattr(_i, k)
            if isinstance(v, Sub): 
                v = v.d
            _d[k] = v
        return _d

    @property
    def _sub_cls(_i):
        return [v for v in _i.__dict__.values() if isinstance(v, Sub)]

    def partial(_i, f:Callable, **kw):
        # _i._debug_print(on=False)
        d = flat_any(_i.d)
        d_k = inspect.signature(f.__init__).parameters.keys()
        d = {k:copy(v) for k,v in d.items() if k in d_k}
        d = {k:v for k,v in d.items() if k in d_k} | kw
        return f(**d)


    def merge(_i, d:dict):
        for k,v in d.items():
            for cls in [_i]+_i._sub_cls:
                if k in cls.__class__.__dict__:
                    try:
                        setattr(cls, k, copy(v))
                    except Exception as e:
                        print(e, '\n Unmerged {k}')
            

    def _to_wandb(_i, d:dict, parent='', sep='.', _l:list=[])->dict:
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

    def _debug_print(_i, on=False, cls=True):
        if on:
            for k,v in vars(_i).items():
                if not k.startswith('_'):
                    print(k, getattr(_i, k))    
            if cls:
                [print(k,v) for k,v in vars(_i.__class__).items() if not k.startswith('_')]



        

