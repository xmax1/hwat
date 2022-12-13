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
import numpy as np

from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum, iterate_folder
from utils import flat_dict, mkdir, cmd_to_dict, dict_to_wandb
from utils import Sub

from user import server, sbatch_cmd, wandb_entity

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'

class Pyfig:

    project_root:   str     = Path().home()/'projects'
    project:        str     = 'hwat'

    data_dir:       Path    = project_root / 'data'  # not a property, might change
    run_path:       Path    = property(lambda _: _.project_path / 'run.py')
    exp_name:       str     = 'demo-final'
    exp_id:         str     = gen_alphanum(n=7)
    
    
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
        n_u:        int         = property(lambda _: (spin + _.n_e)//2 )
        n_d:        int         = property(lambda _: _.n_e - _.n_u)
        
        n_b:        int         = 512
        n_corr:     int         = 20
        n_equil:    int         = 10000  # total number: n_equil_loop = equil_len / corr_len
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
        name            = 'sweep'
        metrics         = dict(
            goal        = 'minimize',
            name        = 'validation_loss',
        )
        parameters = dict(
            batch_size  = {'values' : [16, 32, 64]},
            # epoch       = {'values' : [5, 10, 15]},
            lr          = {'max'    : 0.1, 'min': 0.0001},
        )
        n_sweep         = reduce(
            lambda i,j:i*j,[len(v['values']) for k,v in parameters.items() if 'values' in v])+1
        sweep_id = ''

    class wandb_c(Sub):
        job_type        = 'training'
        entity          = 'hwat'
        wandb_run_path  = ''

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

    def __init__(ii,args:dict={},cap=40,sweep=False,wandb_mode='online',get_sys_arg=True):
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=ii)
                setattr(ii, k, v)
        
        ii.merge(cmd_to_dict(sys.argv[1:], ii.d))
        ii.merge(args)

        mkdir(ii.exp_path)
        
        run = wandb.init(
            job_type    = ii.wandb_c.job_type,
            entity      = ii.wandb_c.entity,
            project     = ii.project,
            dir         = './dump', # ii.exp_path,
            config      = dict_to_wandb(ii.d, ignore=ii._wandb_ignore),
            mode        = wandb_mode,
            settings = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
        )

        if ii.submit_state > 0:
            n_job_running = run_cmds([f'squeue -u {ii.user} -h -t pending,running -r | wc -l'])
            if n_job_running > cap:
                exit(f'There are {n_job_running} on the submit cap is {cap}')
            print(f'{n_job_running} on the cluster')

            _slurm = Slurm(**ii.slurm.d)
            n_run, ii.submit_state = ii.submit_state, 0            
            for _ in range(n_run):
                _slurm.sbatch(ii.slurm.sbatch \
                    + f'out_dir={(mkdir(ii.exp_path/"out"))} {ii.cmd} | tee $out_dir/py.out date "+%B %V %T.%3N" ')

        ii.wandb_c.wandb_run_path = run.path  
        print('run: ', run.path, '✅')

    @property
    def cmd(ii,):
        d = flat_dict(ii.d)
        return ' '.join([f' --{k}  {str(v)} ' for k,v in d.items()])

    @property
    def d(ii, ignore=['d', 'cmd', 'submit', 'partial', 'sweep']):
        out = {}
        for k,v in ii.__class__.__dict__.items():
            if k.startswith('_') or k in ignore:
                continue
            if isinstance(v, partial):
                v = copy(ii.__dict__[k]) # ‼ 🏳 Danger zone - partials may not be part of dict
            else:
                v = getattr(ii, k)
            if isinstance(v, Sub): 
                v = v.d
            out[k] = v
        return out

    @property
    def _sub_cls(ii):
        return [v for v in ii.__dict__.values() if isinstance(v, Sub)]
    
    def submit(ii, sweep=False, commit_msg=None, commit_id=None):
        commit_msg = commit_msg or ii.exp_id
        
        ii.submit_state *= -1
        if ii.submit_state > 0:
            if sweep:
                ii.sweep_id = wandb.sweep(
                    env     = f'conda activate {ii.env};',
                    sweep   = ii.sweep.d, 
                    program = ii.run_path,
                    project = ii.project,
                    name    = ii.exp_name,
                    run_cap = ii.sweep.n_sweep
                )
                ii.submit_state *= ii.sweep.n_sweep
            
            local_out = run_cmds(['git add .', f'git commit -m {commit_msg}', 'git push'], cwd=ii.project_path)
            cmd = f'python -u {ii.run_path} ' + (commit_id or ii.commit_id) + ii.cmd
            server_out = run_cmds_server(ii.server, ii.user, cmd, cwd=ii.server_project_path)

    def partial(ii, f:Callable, get_dict=False, **kw):
        d = flat_any(ii.d)
        d_k = inspect.signature(f.__init__).parameters.keys()
        d = {k:copy(v) for k,v in d.items() if k in d_k}
        d = {k:v for k,v in d.items() if k in d_k} | kw
        if get_dict:
            return d
        return f(**d)

    def merge(ii, d:dict):
        for k,v in d.items():
            for cls in [ii]+ii._sub_cls:
                if k in cls.__class__.__dict__:
                    try:
                        setattr(cls, k, copy(v))
                    except Exception:
                        print(f'Unmerged {k}')
            
    def _debug_print(ii, on=False, cls=True):
        if on:
            for k in vars(ii).keys():
                if not k.startswith('_'):
                    print(k, getattr(ii, k))    
            if cls:
                [print(k,v) for k,v in vars(ii.__class__).items() if not k.startswith('_')]

    @property
    def commit_id(ii,)->str:
        process = run_cmds(['git log --pretty=format:%h -n 1'], cwd=ii.project_path)[0]
        return process.stdout.decode('utf-8')  



        
