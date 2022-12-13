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

from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum
from utils import flat_dict, mkdir, cmd_to_dict, dict_to_wandb
from utils import Sub

from _user import user

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'

class Pyfig:

    project_root:   str     = Path('~/projects')
    project:        str     = 'hwat'
    run_dir:       Path     = Path(__file__).parent.relative_to(Path().home())
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
        n_u:        int         = property(lambda _: (_.spin + _.n_e)//2 )
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

    class wandb_c(Sub):
        job_type        = 'training'
        entity          = 'hwat'
        wandb_run_path  = ''

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
        sbatch          = property(lambda _: 
            f""" 
            module purge
            source ~/.bashrc 
            module load GCC 
            module load CUDA/11.4.1 
            module load cuDNN/8.2.2.26-CUDA-11.4.1 
            conda activate {_._p.env} 
            export MKL_NUM_THREADS=1 
            export NUMEXPR_NUM_THREADS=1 
            export OMP_NUM_THREADS=1 
            export OPENBLAS_NUM_THREADS=1
            pwd
            nvidia-smi
            mv_cmd = f'mv {_._p.TMP}/o-$SLURM_JOB_ID.out {_._p.TMP}/e-$SLURM_JOB_ID.err $out_dir'
            out_dir={(mkdir(_._p.exp_path/"out"))}
        """)
    
    
    TMP:                Path    = Path('./dump/tmp')
    project_path:       Path    = property(lambda _: _.project_root / _.project)
    server_project_path:Path    = property(lambda _: _.project_path)
    n_device:           int     = property(lambda _: count_gpu())
    exp_path:           Path    = property(lambda _: _.project_path/'exp'/_.exp_name/(_.exp_id + _.sweep_id))

    user:               str     = user             # SERVER
    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        
    env:                str     = 'dex'                 # CONDA ENV
    
    _n_submit:          int  = -1                  # #_n_submit-state-flow
    _sys_arg:           list = sys.argv[1:]
    _wandb_ignore:      list = ['sbatch',]

    def __init__(ii, arg:dict={}, cap=3, wandb_mode='online', submit=False, sweep=False, notebook=False):
        ii._input_arg = arg 
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=ii)
                setattr(ii, k, v)
        
        sys_arg = cmd_to_dict(sys.argv[1:], ii.d)
        ii.merge(ii._input_arg | sys_arg)
        
        
        """             |        submit         |       
                        |   True    |   False   | 
                        -------------------------
                    <0  |   server  |   init    |
        _n_submit   =0  |   init    |   NA      |
                    >0  |   slurm   |   NA      |
        """
        run_init_local = (not submit) and (ii._n_submit < 0)
        run_init_cluster = submit and (ii._n_submit == 0)
        run_slurm = submit and (ii._n_submit > 0)
        run_server = submit and (ii._n_submit < 0)
        
        if run_init_local or run_init_cluster:
            print('Running __init__')
            run = wandb.init(
                    job_type    = ii.wandb_c.job_type,
                    entity      = ii.wandb_c.entity,
                    project     = ii.project,
                    dir         = mkdir(ii.exp_path),
                    config      = dict_to_wandb(ii.d, ignore=ii._wandb_ignore),
                    mode        = wandb_mode,
                    settings    = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
                )
            
            ii.wandb_c.wandb_run_path = run.path  
            
            if ii.sweep_id:
                wandb.agent(ii.sweep_id, count=1)
        
            ii.log(ii.d, create=True)
            
        if run_slurm:
            print('Submitting runs to slurm')
            n_job_running = run_cmds([f'squeue -u {ii.user} -h -t pending,running -r | wc -l'])
            if n_job_running < cap:        
                for sub in range(1, ii._n_submit+1):
                    Slurm(**ii.slurm.d).sbatch(ii.slurm.sbatch + '\n' + ii._run_cmd + ' --_n_submit 0')
                    print(ii.slurm.sbatch + '\n' + ii._run_cmd + ' --_n_submit 0')
                    if sub > 5:
                        break
            exit(f'{sub} submitted, {n_job_running} on cluster before, cap is {cap}')
        
        if run_server:
            print('sshing to server and running this file')
            if ii._n_submit < 0:
                if sweep or ('sweep' in ii._input_arg):
                    print('Running a sweep')
                    ii.sweep_id = wandb.sweep(
                        env     = f'conda activate {ii.env};',
                        sweep   = ii.sweep.d, 
                        program = ii.run_dir / ii.run_name,
                        project = ii.project,
                        name    = ii.exp_name,
                        run_cap = ii.sweep.n_sweep
                    )
                
                local_out = run_cmds(['git add .', f'git commit -m "run_things"', 'git push origin main'], cwd=ii.project_path)
                print(local_out)
                print(ii.server, ii.user, ii.server_project_path)
                git_cmd = 'git pull origin main'
                run_cmd = f'python {str(ii.run_name)} {ii.cmd}'
                server_out = run_cmds_server(ii.server, ii.user, git_cmd, ii.server_project_path)[0]
                print(server_out)
                server_out = run_cmds_server(ii.server, ii.user, run_cmd, ii.run_dir)[0]
                print(server_out)
                exit('Submitted to server.')
            
    @property
    def cmd(ii, ignore=['sbatch', ]):
        d = flat_dict(ii.d)
        # to_cmd_string = lambda v: str(v).replace('\n', '-CR-').replace(' ', '-WS-')
        to_cmd_string = lambda v: str(v)
        cmd_d = {str(k).replace(" ", ""):to_cmd_string(v).replace(" ", "") for k,v in d.items() if not k in ignore}
        return ' '.join([f'--{k} {v}' for k,v in cmd_d.items() if v])

    @property
    def d(ii, ignore=['d', 'cmd', 'submit', 'partial', 'sweep', 'save', 'load', 'log', 'merge']):
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
            for cls in [ii,] + ii._sub_cls:
                if k in cls.__class__.__dict__:
                    if isinstance(cls.__class__.__dict__[k], property):
                        continue
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


    
    


import pickle as pk


file_interface_all = dict(
    pk = dict(
        r = pk.load,
        w = pk.dump,
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

| tee $out_dir/py.out date "+%B %V %T.%3N


"""