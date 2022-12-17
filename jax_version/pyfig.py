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
from time import sleep


from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum
from utils import flat_dict, mkdir, cmd_to_dict, dict_to_wandb, iterate_n_dir
from utils import type_me
from utils import Sub

from _user import user

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'

class Pyfig:
    # SUB CLASSES CANNOT CALL EACH OTHER

    run_name:       Path    = 'run.py'
    
    exp_name:       str     = 'demo-final'
    sweep_id_code:  str     = ''
    run_id:         str     = gen_alphanum(n=7)
    
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
        name            = 'demo'
        method          = 'grid'
        parameters = dict(
            n_b  = {'values' : [16, 32, 64]},
        )

    class wandb_c(Sub):
        job_type        = 'training'
        entity          = property(lambda _: _._p.project)
        name            = property(lambda _: _._p.exp_name)
        program         = property(lambda _: _._p.run_dir/_._p.run_name)
        
    class slurm(Sub):
        mail_type       = 'FAIL'
        partition       ='sm3090'
        nodes           = 1                # n_node
        ntasks          = 8                # n_cpu
        cpus_per_task   = 1     
        time            = '0-12:00:00'     # D-HH:MM:SS
        gres            = 'gpu:RTX3090:1'
        output          = property(lambda _: _._p.exp_path /'o-%j.out')
        error           = property(lambda _: _._p.exp_path /'e-%j.err')
        job_name        = property(lambda _: _._p.exp_name)  # this does not call the instance it is in
    
    TMP:                Path    = mkdir(Path('./dump/tmp'))
    _home:              Path    = property(lambda _: Path().home())
    project:            str     = property(lambda _: 'hwat')
    run_dir:            Path    = property(lambda _: Path(__file__).parent.relative_to(_._home))
    project_dir:        Path    = property(lambda _: (_._home / 'projects' / _.project))
    server_project_dir: Path    = property(lambda _: _.project_dir.relative_to(_._home))
    exp_id:             str     = property(lambda _: _.run_id + _.sweep_id_code)
    iterate_state:      bool    = True
    exp_path:           Path    = \
        property(lambda _: iterate_n_dir(Path('exp')/_.exp_name, _._single_use_switch('iterate_state'))/_.exp_id)
    sweep_id:       str         = property(lambda _: (f'{_.wandb_c.entity}/{_.project}/{_.sweep_id_code}')*bool({_.sweep_id_code}))
        
    n_device:           int     = property(lambda _: count_gpu())
    run_sweep:          bool    = False
    user:               str     = user             # SERVER
    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        
    env:                str     = 'dex'                 # CONDA ENV
    
    n_job:              int  = -1                  # #n_job-state-flow
    _run_cmd:           str  = property(lambda _: f'python {str(_.run_name)} \
                                            {_.cmd*(~bool(_.sweep_id_code)) + _.wandb_cmd*bool(_.sweep_id_code)}')
    
    _git_commit_cmd:    list = ['git commit -a -m "run_things"', 'git push origin main']
    _git_pull_cmd:      list = ['git fetch --all', 'git reset --hard origin/main']
    _sys_arg:           list = sys.argv[1:]
    _wandb_ignore:      list = ['d', 'cmd', 'partial', 'save', 'load', 'log', 'merge'] + ['sbatch', 'sweep']
    
    _useful = 'ssh amawi@svol.fysik.dtu.dk "killall -9 -u amawi"'
    
    def __init__(ii, arg:dict={}, cap=3, wandb_mode='online', submit=False, run_sweep=False): 
        arg.update(dict(run_sweep=run_sweep))
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=ii)
                setattr(ii, k, v)
        
        sys_arg = cmd_to_dict(sys.argv[1:], flat_any(ii.d))
        print(sys_arg)

        arg = arg | sys_arg
        ii.merge(arg)
        
        """             |        submit         |       
                        |   True    |   False   | 
                        -------------------------
                    <0  |   server/init  |   init    |
        n_job       =0  |   init         |   NA      |
                    >0  |   slurm        |   NA      | """
        
        run_init_local = (not submit) and (ii.n_job < 0)
        run_init_cluster = submit and (ii.n_job == 0)
        
        # sweep local: init -> sweep 
        # sweep server: slurm 
        # sweep cluster: init -> agent
        
        # run local: init
        
        if run_init_local or run_init_cluster:
            ii.log(dict(init=dict(sweep_id=ii.sweep_id)), create=True)
            if ii.sweep_id:
                wandb.init()
                wandb.agent(sweep_id=ii.sweep_id, count=1)
            else:
                run = wandb.init(
                    entity      = ii.wandb_c.entity,  # team name is hwat
                    project     = ii.project,         # sub project in team
                    dir         = ii.exp_path,
                    config      = dict_to_wandb(ii.d, ignore=ii._wandb_ignore),
                    mode        = wandb_mode,
                    settings    = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
                    id          = ii.exp_id
                )    
        
        if submit and ii.n_job > 0 and ii._n_job_running < cap:
            n, ii.n_job  = ii.n_job, 0
            ii.log(dict(slurm_init=dict(sbatch=ii.sbatch, run_cmd=ii._run_cmd, n_job=ii.n_job)), create=True, log_name='slurm_init.log')
            for sub in range(1, n+1):
                Slurm(**ii.slurm.d).sbatch(ii.sbatch + '\n' + ii._run_cmd)
                sleep(run_sweep*3)
            sys.exit(f'Submitted {sub} to slurm')

        if submit and ii.n_job < 0: 
            if run_sweep:
                ii.sweep_id_code = wandb.sweep(
                    sweep   = ii.sweep.d | dict(name=ii.wandb_c.name), 
                    project = ii.project,
                    entity  = ii.wandb_c.entity,
                )
                n_step_grid = [len(v['values']) for k,v in ii.sweep.parameters.items() if 'values' in v]
                
            ii.n_job = reduce(lambda a,b: a*b, n_step_grid if run_sweep else [1])
            
            ii.log(dict(server_init=dict(run_cmd=ii._run_cmd, n_job=ii.n_job)), create=True, log_name='server_init.log')
            
            run_cmds(ii._git_commit_cmd, cwd=ii.project_dir)
            run_cmds_server(ii.server, ii.user, ii._git_pull_cmd, ii.server_project_dir)
            run_cmds_server(ii.server, ii.user, ii._run_cmd, ii.run_dir)
        
            print(f'Go to https://wandb.ai/{ii.wandb_c.entity}/{ii.project}/{ii.exp_id}')
            sys.exit(f'Submitted {ii.n_job} to server')
    
    def _single_use_switch(ii, k):
        """ if True, turn False, return True
        if False, keep False, return False """
        state = getattr(ii, k)
        setattr(ii, k, False)
        return state
        
    @property
    def sbatch(ii,):
        s = f"""\
        module purge
        source ~/.bashrc
        module load GCC
        module load CUDA/11.4.1
        module load cuDNN/8.2.2.26-CUDA-11.4.1
        conda activate {ii.env}"""
        # out_dir={mkdir(ii.exp_path)}"""
        # mv_cmd="mv {ii.TMP}/o-$SLURM_JOB_ID.out {ii.TMP}/e-$SLURM_JOB_ID.err $out_dir"
        # echo $mv_cmd
        return '\n'.join([' '.join(v.split()) for v in s.split('\n')])
    
    @property
    def cmd(ii):
        d = flat_dict(get_cls_dict(ii, sub_cls=True, ignore=['sweep',]))
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
        cmd_d = {str(k).replace(" ", ""): str(v).replace(" ", "") for k,v in d.items()}
        return ' '.join([f'--{k} {v}' for k,v in cmd_d.items() if v])
    
    @property
    def wandb_cmd(ii):
        d = flat_dict(get_cls_dict(ii, sub_cls=True, ignore=['sweep',] + list(ii.sweep.parameters.keys())))
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
        cmd_d = {str(k).replace(" ", ""): str(v).replace(" ", "") for k,v in d.items()}
        return ' '.join([f' --{k}={v}' for k,v in cmd_d.items() if v])
    
        cmd = ' ' + ii.cmd
        cmd = [x.strip() for x in cmd.split(' --')][1:]
        cmd = [x.split(' ', maxsplit=1) for x in cmd]
        new_cmd = ''
        for c in cmd:
            new_cmd += ' --'+c[0].replace(' ', '')
            if len(c) == 2:
                new_cmd += f'={c[1].replace(" ", "")}'
        return cmd

    @property
    def d(ii):
        return get_cls_dict(ii, sub_cls=True, prop=True)

    @property
    def commit_id(ii,)->str:
        process = run_cmds(['git log --pretty=format:%h -n 1'], cwd=ii.project_dir)[0]
        return process.stdout.decode('utf-8') 
    
    @property
    def _n_job_running(ii,):
        return len(run_cmds([f'squeue -u amawi -t pending,running -h -r'], cwd='.')[0].stdout.decode('utf-8'))
    
    def partial(ii, f:Callable, d=None, get_d=False, print_d=False, **kw):
        d = flat_any(d if d else ii.d)
        d_k = inspect.signature(f.__init__).parameters.keys()
        d = {k:copy(v) for k,v in d.items() if k in d_k}
        d = {k:v for k,v in d.items() if k in d_k} | kw
        if get_d:
            return d
        if print_d:
            pprint.pprint(d)
        return f(**d)

    def merge(ii, merge:dict):
        for k,v in merge.items():
            assigned = False
            for cls in [ii,] + list(ii._sub_cls.values()):
                ref = get_cls_dict(cls,)
                if k in ref:
                    # print(v, ref[k], type(v), type(ref[k]))
                    v = type_me(v, ref[k])
                    try:
                        setattr(cls, k, copy(v))
                        assigned = True
                    except Exception:
                        print(f'Unmerged {k} at setattr')
            if not assigned:
                print(k, v, 'not assigned')
                        
    @property
    def _sub_cls(ii) -> dict:
        return {k:v for k,v in ii.__dict__.items() if isinstance(v, Sub)}
    
    def save(ii, data, file_name):
        path:Path = ii.exp_path / file_name
        data_save_load(path, data)
        
    def load(ii, file_name):
        path:Path = ii.exp_path / file_name
        assert path.suffix == 'pk'
        data_save_load(path)
        
    def log(ii, info: dict, create=False, log_name='log.out'):
        mkdir(ii.exp_path)
        mode = 'w' if create else 'a'
        info = pprint.pformat(info)
        for p in ['log.tmp', ii.exp_path/log_name]:
            with open(p, mode) as f:
                f.writelines(info)

def get_cls_dict(
        cls,
        ref:list|dict=None,
        sub_cls=False, 
        fn=False, 
        prop=False, 
        hidn=False,
        ignore:list=None
    ) -> dict:
        ignore = ['d', 'cmd', 'partial', 'save', 'load', 'log', 'merge'] + (ignore or [])
    
        items = []
        for k, v_cls in cls.__class__.__dict__.items():
            
            is_builtin = k.startswith('__')
            should_ignore = k in ignore
            not_in_ref = k in (ref if ref else [])
            
            if not (is_builtin or should_ignore or not_in_ref):
                
                if (not hidn) and k.startswith('_'):
                    continue
                    
                is_fn = (not fn) and isinstance(v_cls, partial)
                if is_fn:
                    continue
                
                is_prop = (not prop) and isinstance(v_cls, property)
                if is_prop:
                    continue
                
                v = getattr(cls, k)
                
                if isinstance(v, Sub):
                    if sub_cls:
                        sub_d = get_cls_dict(v,
                            ref=ref, sub_cls=False, fn=fn, prop=prop, hidn=hidn, ignore=ignore)
                        items.append([k, sub_d])
                    continue
                
                items.append([k, v])
        return dict(items)




def cls_filter(
    cls, k: str, v, 
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