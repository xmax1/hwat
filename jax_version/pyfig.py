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
    _home:              Path    = property(lambda _: Path().home())
    project:            str     = property(lambda _: 'hwat')
    run_dir:            Path    = property(lambda _: Path(__file__).parent.relative_to(_._home))
    project_dir:        Path    = property(lambda _: (_._home / 'projects' / _.project))
    server_project_dir: Path    = property(lambda _: _.project_dir.relative_to(_._home))
    exp_path:           Path    = property(lambda _: _.run_dir/'exp'/_.exp_name/(_.exp_id + _.sweep_id))
        
    n_device:           int     = property(lambda _: count_gpu())

    user:               str     = user             # SERVER
    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        
    env:                str     = 'dex'                 # CONDA ENV
    
    _run_cmd:           str  = property(lambda _: f'python {str(_.run_name)} {_.cmd}')
    n_submit:          int  = -1                  # #n_submit-state-flow
    _sys_arg:           list = sys.argv[1:]
    _wandb_ignore:      list = ['sbatch',]

    def __init__(ii, arg:dict={}, cap=3, wandb_mode='online', submit=False, sweep=False, notebook=False):
        ii._input_arg = arg 
        
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=ii)
                setattr(ii, k, v)
        
        sys_arg = cmd_to_dict(sys.argv[1:], flat_any(ii.d))
        ii.merge(ii._input_arg | sys_arg)
        
        
        """             |        submit         |       
                        |   True    |   False   | 
                        -------------------------
                    <0  |   server  |   init    |
        n_submit   =0  |   init    |   NA      |
                    >0  |   slurm   |   NA      |
        """
        run_init_local = (not submit) and (ii.n_submit < 0)
        run_init_cluster = submit and (ii.n_submit == 0)
        run_slurm = submit and (ii.n_submit > 0)
        run_server = submit and (ii.n_submit < 0)
        
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
            n_job_running = len(run_cmds([f'squeue -u amawi -t pending,running -h -r'], cwd='.')[0].stdout.decode('utf-8'))
            ii.log({'n_job_running': n_job_running})
            if n_job_running < cap:    
                ii.n_submit = 0
                for sub in range(1, ii.n_submit+1):
                    Slurm(**ii.slurm.d).sbatch(ii.slurm.sbatch + '\n' + ii._run_cmd )
                    ii.log({'slurm': ii.slurm.sbatch + '\n' + ii._run_cmd })
                    if sub > 5:
                        break
            exit(f'{sub} submitted, {n_job_running} on cluster before, cap is {cap}')
        
        if run_server:
            print('sshing to server and running this file')
            if ii.n_submit < 0:
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
                
                local_out = run_cmds(['git commit -a -m "run_things"', 'git push origin main'], cwd=ii.project_dir)
                print(local_out)
                git_cmd = ['git fetch --all', 'git reset --hard origin/main']
                # git_cmd = 'git pull origin main --force'
                
                server_out = run_cmds_server(ii.server, ii.user, git_cmd, ii.server_project_dir)[0]
                print(server_out)
                
                ii.n_submit = max(1, ii.sweep.n_sweep)
                cmd = ii._run_cmd
                print(cmd)
                server_out = run_cmds_server(ii.server, ii.user, cmd, ii.run_dir)[0]
                print(server_out)
                exit('Submitted to server.')
            
    @property
    def cmd(ii):
        d = flat_dict(ii._get_dict(get_prop=False))
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
        cmd_d = {str(k).replace(" ", ""): str(v).replace(" ", "") for k,v in d.items()}
        return ' '.join([f'--{k} {v}' for k,v in cmd_d.items() if v])

    @property
    def d(ii):
        return ii._get_dict(get_prop=True)

    @property
    def _sub_cls(ii):
        return [v for v in ii.__dict__.values() if isinstance(v, Sub)]
    
    def _get_dict(ii, cls=None, get_prop=True, _ignore=['sweep', 'sbatch']):
        ignore = ['d', 'cmd', 'partial', 'save', 'load', 'log', 'merge'] + _ignore
        cls = ii if cls is None else cls
        items = []
        for k, v_cls in cls.__class__.__dict__.items():
            if k.startswith('_'):
                continue
            if k in ignore:
                continue
            
            v = getattr(cls, k)
            
            if isinstance(v, partial):
                continue # v = copy(ii.__dict__[k])
            if isinstance(v_cls, property):
                if not get_prop:
                    continue
                
            if isinstance(v, Sub):
                items.extend(ii._get_dict(cls=v).items())
            else:
                items.append((k, v))
        return dict(items)
        

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
        process = run_cmds(['git log --pretty=format:%h -n 1'], cwd=ii.project_dir)[0]
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