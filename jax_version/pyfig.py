from utils import flat_any
import inspect
from typing import Callable
from functools import reduce, partial
from simple_slurm import Slurm
import wandb
from pathlib import Path
import sys
import pprint
from copy import copy
import numpy as np
import re


from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum
from utils import flat_dict, mkdir, cmd_to_dict, dict_to_wandb
from utils import type_me
from utils import Sub

from _user import user

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'

class Pyfig:

    run_name:       Path    = 'run.py'
    
    exp_name:       str     = 'demo-final'
    exp_id:         str     = gen_alphanum(n=7)
    sweep_id:       str     = ''
    
    seed:           int     = 808017424 # grr
    dtype:          str     = 'float32'
    n_step:         int     = 200
    log_metric_step:int     = 100
    log_state_step: int     = 10          
	
    class data(Sub):
        """
        n_e = \sum_i charge_nuclei_i - charge = n_e
        spin = n_u - n_d
        n_e = n_u + n_d
        n_u = 1/2 ( spin + n_e ) = 1/2 ( spin +  \sum_i charge_nuclei_i - charge)
        """
        charge:     int         = 0
        spin:       int         = 0
        a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
        a_z:        np.ndarray  = np.array([4.,])

        n_e:        int         = property(lambda _: int(sum(_.a_z)))
        n_u:        int         = property(lambda _: (_.spin + _.n_e)//2)
        n_d:        int         = property(lambda _: _.n_e - _.n_u)
        
        n_b:        int         = 512
        n_corr:     int         = 20
        n_equil:    int         = 10000
        acc_target: int         = 0.5

    class model(Sub):
        with_sign:      bool    = False
        n_sv:           int     = 32
        n_pv:           int     = 16
        n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)
        n_fb:           int     = 2
        n_det:          int     = 1
        terms_s_emb:    list    = ['ra', 'ra_len']
        terms_p_emb:    list    = ['rr', 'rr_len']

    class sweep(Sub):
        method          = 'random'
        metrics         = dict(goal = 'minimize', name = 'validation_loss')
        parameters = dict(
            n_b  = {'values' : [16, 32, 64]},
            lr   = {'max'    : 0.1, 'min': 0.0001},
        )
        n_sweep         = reduce(
            lambda i,j:i*j,[len(v['values']) for k,v in parameters.items() if 'values' in v])+1

    class wandb_c(Sub):
        run             = None
        job_type        = 'training'
        entity          = property(lambda _: _._p.project)
        name            = property(lambda _: _._p.exp_name)
        program         = property(lambda _: _._p.run_dir/_._p.run_name)
        project         = property(lambda _: _._p.project)
        run_cap         = property(lambda _: _._p.sweep.n_sweep)
        sweep           = property(lambda _: _._p.sweep.d)
        wandb_mode      = 'disabled',
        mode            = property(lambda _: _.wandb_mode)
        settings        = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
        config          = property(lambda _: dict_to_wandb(_._i.d, ignore=_._p._wandb_ignore))
        
    class slurm(Sub):
        mail_type       = 'FAIL'
        partition       ='sm3090'
        nodes           = 1                # n_node
        ntasks          = 8                # n_cpu
        cpus_per_task   = 1     
        time            = '0-12:00:00'     # D-HH:MM:SS
        gres            = 'gpu:RTX3090:1'
        output          = property(lambda _: _._p.TMP /'o-%j.out')
        error           = property(lambda _: _._p.TMP /'e-%j.err')
        job_name        = property(lambda _: _._p.exp_name)  # this does not call the instance it is in
    
    sbatch: str = property(lambda _: 
    f""" 
    module purge
    source ~/.bashrc 
    module load GCC 
    module load CUDA/11.4.1 
    module load cuDNN/8.2.2.26-CUDA-11.4.1 
    conda activate {_.env} 
    mv_cmd = f'mv {_.TMP}/o-$SLURM_JOB_ID.out {_.TMP}/e-$SLURM_JOB_ID.err $out_dir'
    out_dir={(mkdir(_.exp_path/"out"))}
    """)
    
    TMP:                Path    = Path('./dump/tmp')
    _home:              Path    = property(lambda _: Path().home())
    project:            str     = property(lambda _: 'hwat')
    run_dir:            Path    = property(lambda _: Path(__file__).parent.relative_to(_._home))
    project_dir:        Path    = property(lambda _: (_._home / 'projects' / _.project))
    server_project_dir: Path    = property(lambda _: _.project_dir.relative_to(_._home))
    exp_path:           Path    = property(lambda _: (_.run_dir/'exp'/_.exp_name/(_.exp_id + _.sweep_id)).relative_to('.'))
        
    n_device:           int     = property(lambda _: count_gpu())

    user:               str     = user             # SERVER
    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        
    env:                str     = 'dex'                 # CONDA ENV
    
    n_job:           int  = -1                  # #n_job-state-flow
    _run_cmd:           str  = property(lambda _: f'python {str(_.run_name)} "{_.cmd}"')
    _sys_arg:           list = sys.argv[1:]
    _ignore:            list = ['d', 'cmd', 'partial', 'save', 'load', 'log', 'merge']
    _wandb_ignore:      list = _ignore + ['sbatch', 'sweep']
    
    def __init__(ii, arg:dict={}, cap=3, wandb_mode='online', submit=False, sweep=False): 
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=ii)
                setattr(ii, k, v)
        
        ii._input_arg = arg | cmd_to_dict(sys.argv[1:], flat_any(ii.d))
        ii.merge(ii._input_arg)
        mkdir(ii.exp_path)
        
        ii.log(ii.d, create=True)
        
        """             |        submit         |       
                        |   True    |   False   | 
                        -------------------------
                    <0  |   server  |   init    |
        n_job       =0  |   init    |   NA      |
                    >0  |   slurm   |   NA      | """
        
        run_init_local = (not submit) and (ii.n_job < 0)
        run_init_cluster = submit and (ii.n_job == 0)
        if run_init_local or run_init_cluster:
            ii.wandb_c.run = wandb.init(**ii.wandb_c.d)
            wandb.agent(ii.sweep_id, count=1)
        
        if submit:
            if ii.n_job > 0 and ii.n_job_running < cap:
                n, ii.n_job  = ii.n_job, 0
                for sub in range(1, n+1):
                    Slurm(**ii.slurm.d).sbatch(ii.slurm.sbatch + '\n' + ii._run_cmd)
                ii.log({'slurm': ii.slurm.sbatch + '\n' + ii._run_cmd})
            
            if ii.n_job < 0: 
                if sweep or ('sweep' in arg):
                    ii.sweep_id = wandb.sweep(**ii.wandb_c.d)
                
                _git_commit_cmd = ['git commit -a -m "run_things"', 'git push origin main']
                _git_pull_cmd = ['git fetch --all', 'git reset --hard origin/main']
                
                run_cmds(_git_commit_cmd, cwd=ii.project_dir)
                run_cmds_server(ii.server, ii.user, _git_pull_cmd, ii.server_project_dir)
                
                ii.n_job = max(1, ii.sweep.n_sweep*sweep)
                run_cmds_server(ii.server, ii.user, ii._run_cmd, ii.run_dir)
            
            # ii.log()    
            sys.exit(f'Submitted {ii.n_job}{(sub+1)} to {(ii.n_job < 0)*"server"} {(ii.n_job > 0)*"slurm"}')
        
    @property
    def cmd(ii):
        d = flat_dict(ii._get_dict(get_prop=False))
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
        cmd_d = {str(k).replace(" ", ""): str(v).replace(" ", "") for k,v in d.items()}
        return ' '.join([f'--{k} {v}' for k,v in cmd_d.items() if v])

    @property
    def d(ii):
        return ii._get_cls_dict(ii)

    @property
    def commit_id(ii,)->str:
        process = run_cmds(['git log --pretty=format:%h -n 1'], cwd=ii.project_dir)[0]
        return process.stdout.decode('utf-8') 
    
    @property
    def n_job_running(ii,):
        return len(run_cmds([f'squeue -u amawi -t pending,running -h -r'], cwd='.')[0].stdout.decode('utf-8'))
    
    def partial(ii, f:Callable, get_dict=False, **kw):
        d = flat_any(ii.d)
        d_k = inspect.signature(f.__init__).parameters.keys()
        d = {k:copy(v) for k,v in d.items() if k in d_k}
        d = {k:v for k,v in d.items() if k in d_k} | kw
        if get_dict:
            return d
        return f(**d)

    def merge(ii, merge:dict):
        for k,v in merge.items():
            for cls in [ii,] + ii._sub_cls:
                ref = cls.__class__.__dict__
                if cls_filter(v, cls=cls, ref=ref):
                    v = type_me(v, v_ref=ref[k])
                    try:
                        setattr(cls, k, copy(v))
                    except Exception:
                        print(f'Unmerged {k}')
                        
    @property
    def _sub_cls(ii):
        return {k:v for k,v in ii.__dict__.items() if isinstance(v, Sub)}
    
    def _get_cls_dict(ii, cls, ignore=None):
        cls_d = cls.__class__.__dict__
        items = []
        for k,v in cls_d.items():
            if cls_filter(cls, is_prop=True, is_fn=True, is_sub=True, ref=cls_d, ignore=ignore):
                if isinstance(v, Sub):
                    items.extend(ii._get_cls_dict(v, ignore=ignore).items())
                else:
                    items += (k, ii._get_from_cls(k, cls))  
        return dict(items)
    
    @staticmethod
    def _get_from_cls(cls, k, v):
        if isinstance(v, partial):
            return copy(cls.__dict__[k])
        return getattr(cls, k)

    def save(ii, data, file_name):
        path:Path = ii.exp_path / file_name
        data_save_load(path, data)
        
    def load(ii, file_name):
        path:Path = ii.exp_path / file_name
        assert path.suffix == 'pk'
        data_save_load(path)
        
    def log(ii, info: dict, create=False):
        mode = 'w' if create else 'a'
        info = pprint.pformat(info)
        for p in ['log.tmp', ii.exp_path/'log.out']:
            with open(p, mode) as f:
                f.writelines(info)
    
def cls_filter(
    cls, k, v, 
    ref:list|dict=None,
    is_fn=False, 
    is_sub=False, 
    is_prop=False, 
    is_hidn=False,
    ignore:list=None,
    keep = False,
):  
    
    is_builtin = k.startswith('__')
    should_ignore = k in (ignore if ignore else [])
    not_in_ref = k in (ref if ref else [k])
    
    if not (is_builtin or should_ignore or not_in_ref):
        keep |= is_hidn and k.startswith('_')
        keep |= is_sub and isinstance(v, Sub)
        keep |= is_fn and isinstance(v, partial)
        keep |= is_prop and isinstance(cls.__class__.__dict__[k], property)
    return keep

# Data Interface

import pickle as pk
import yaml
import json

file_interface_all = dict(
    pk = dict(
        r = pk.load,
        w = pk.dump,
    ),
    yaml = dict(
        r = yaml.load,
        w = yaml.dump,
    ),
    json = dict(
        r = json.load,
        w = json.dump,
    )
)      

def data_save_load(path:Path, data=None):
    file_type = path.suffix
    mode = 'r' if data is None else 'w'
    mode += 'b' if file_type in ['pk',] else ''
    interface = file_interface_all[file_type][mode]
    with open(path, mode) as f:
        data = interface(data, f) if data is not None else interface(f)
    return data


        

""" Bone Zone

export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OMP_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
nvidia-smi`

| tee $out_dir/py.out date "+%B %V %T.%3N


env     = f'conda activate {ii.env};',
# pyfig
def _debug_print(ii, on=False, cls=True):
        if on:
            for k in vars(ii).keys():
                if not k.startswith('_'):
                    print(k, getattr(ii, k))    
            if cls:
                [print(k,v) for k,v in vars(ii.__class__).items() if not k.startswith('_')]

"""