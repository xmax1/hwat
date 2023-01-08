from utils import flat_any
import inspect
from typing import Callable, Union
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
import optree

from utils import run_cmds, run_cmds_server, count_gpu, gen_alphanum
from utils import mkdir, cmd_to_dict, dict_to_wandb, iterate_n_dir
from utils import type_me
from utils import Sub

from dump.user import user

docs = 'https://www.notion.so/5a0e5a93e68e4df0a190ef4f6408c320'

class Pyfig:
    # SUB CLASSES CANNOT CALL EACH OTHER

    run_name:       Path        = 'run.py'
    sweep_id:       str         = ''

    seed:           int         = 808017424 # grr
    dtype:          str         = 'float32'
    n_step:         int         = 10000
    log_metric_step:int         = 10
    log_state_step: int         = 10          
	
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
        
        n_b:        int         = 256
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
        ke_method:      str     = 'vjp'

    class sweep(Sub):        
        # program         = property(lambda _: _._p.run_name)
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
        

# When using --cpus-per-task to run multithreaded tasks, be aware that CPU binding is inherited from the parent of the process. This means that the multithreaded task should either specify or clear the CPU binding itself to avoid having all threads of the multithreaded task use the same mask/CPU as the parent. Alternatively, fat masks (masks which specify more than one allowed CPU) could be used for the tasks in order to provide multiple CPUs for the multithreaded tasks.
    class slurm(Sub):
        # A job consists in one or more steps, each consisting in one or more tasks each using one or more CPU.
        mail_type       = 'FAIL'
        partition       ='sm3090'
        nodes           = '1-1' # (MIN-MAX) 
        cpus_per_task   = 4
        mem_per_cpu     = 1024
        ntasks          = property(lambda _: _._p.n_gpu)
        time            = '0-01:00:00'     # D-HH:MM:SS
        partition       = 'sm3090'
        gres            = property(lambda _: 'gpu:RTX3090:' + str(_._p.n_gpu))
        output          = property(lambda _: _._p.exp_path /'slurm/o-%j.out')
        error           = property(lambda _: _._p.exp_path /'slurm/e-%j.err')
        job_name        = property(lambda _: _._p.exp_name)
        
    #SBATCH --cpus-per-task       4
    #SBATCH --mem-per-cpu         1024
    #SBATCH --error               dump/exp/demo-12/Wvyonoa/slurm/e-%j.err
    #SBATCH --gres                gpu:RTX3090:2
    #SBATCH --job-name            demo
    #SBATCH --mail-type           FAIL
    #SBATCH --nodes               1-1
    #SBATCH --ntasks              2
    #SBATCH --output              dump/exp/demo-12/Wvyonoa/slurm/o-%j.out
    #SBATCH --partition           sm3090
    #SBATCH --time                0-01:00:00
        
    class dist(Sub):
        exchange_dir        = property(lambda _: mkdir(_._p.exp_path / 'exchange'))
        accumulate_step     = 5
        dist_id: str        = property(lambda _: f'{_._p.hostname}-{_.gpu_id}')
        head: bool          = True
        gpu_id: str      = property(lambda _: ''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader')))

    dump:               str     = property(lambda _: Path('dump'))
    TMP:                Path    = mkdir(Path('./dump/tmp'))
    project:            str     = property(lambda _: 'hwat')
    run_dir:            Path    = property(lambda _: Path(__file__).parent.relative_to(Path().home()))
    project_dir:        Path    = property(lambda _: Path().home() / 'projects' / _.project)
    server_project_dir: Path    = property(lambda _: _.project_dir.relative_to(Path().home()))
    exp_id:             str     = gen_alphanum(n=7)
        
    sweep_path_id:      str     = property(lambda _: (f'{_.wandb_c.entity}/{_.project}/{_.sweep_id}')*bool(_.sweep_id))
        
    n_device:           int     = property(lambda _: count_gpu())
    run_sweep:          bool    = False
    user:               str     = 'amawi'           # SERVER
    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        
    env:                str     = 'lumi'                 # CONDA ENV
    
    debug:              bool     = False
    wb_mode:            str      = 'disabled'
    submit:             bool     = False
    cap:                int      = 40
    
    exp_path:           Path     = Path('')
    exp_name:           str      = 'junk'
    n_gpu:              int      = 1  # submission devices
    head:               bool     = False
    
    commit_id           = property(lambda _: run_cmds('git log --pretty=format:%h -n 1', cwd=_.project_dir))
    hostname: str       = property(lambda _: run_cmds('hostname'))
    _n_job_running: int = property(lambda _: len(run_cmds('squeue -u amawi -t pending,running -h -r', cwd='.')))
    
    _not_in_sweep = property(lambda _: \
        get_cls_dict(_, sub_cls=True, ignore=['sweep',]+list(_.sweep.parameters.keys()), to_cmd=True, flat=True)
    )
    
    exe_mode: str = property(lambda _: f'sweep'*_.run_sweep + 'python'*(not _.run_sweep))
    distribute = property(lambda _: bool(_.n_gpu - 1))
    device_type: str = 'cuda'  # rocm
       
    _pci_id:            str      = property(lambda _: ''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader')))
    _git_commit_cmd:    list     = 'git commit -a -m "run"' # !NB no spaces in msg 
    _git_pull_cmd:      list     = ['git fetch --all', 'git reset --hard origin/main']
    _sys_arg:           list     = sys.argv[1:]
    _ignore:            list     = ['d', 'cmd', 'partial', 'save', 'load', 'log', 'merge', 'accumulate']
    _wandb_ignore:      list     = ['sbatch', 'sweep']
    
    _useful = 'ssh amawi@svol.fysik.dtu.dk "killall -9 -u amawi"'
    
    def __init__(ii, arg={}, wb_mode='online', submit=False, run_sweep=False, notebook=False, debug=False, cap=3, **kw):
        # wb_mode: online, disabled, offline 
        print('exe_mode: ', ii.exe_mode)
        print(run_cmds('nvidia-smi --list-gpus'))
        
        init_arg = dict(run_sweep=run_sweep, submit=submit, debug=debug, wb_mode=wb_mode, cap=cap) | kw
        
        print('init sub classes')
        for k,v in Pyfig.__dict__.items():
            if isinstance(v, type):
                v = v(parent=ii)
                setattr(ii, k, v)
        
        print('updating configuration')
        sys_arg = cmd_to_dict(sys.argv[1:], flat_any(ii.d)) if not notebook else {}
        pprint.pprint(sys_arg)
        ii.merge(arg | init_arg | sys_arg)
        ii.log(ii.d, create=True, log_name='post_var_init.log')
        ii.log(ii.sbatch, create=True, log_name='run_cmd.log')
        
        if ii.exp_path == Path(''): 
            if not ii.run_sweep:
                ii.exp_path = iterate_n_dir(ii.dump/'exp'/ii.exp_name, True) / ii.exp_id
            else:
                ii.exp_path = iterate_n_dir(ii.dump/'exp'/ii.exp_name, True) / ii.sweep_id
        mkdir(ii.exp_path / 'slurm')
        print('exp_path:', ii.exp_path)
        
        if not ii.submit:
            print('running script')
            if ii.head:
                ii._run = wandb.init(
                        entity      = ii.wandb_c.entity,  # team name is hwat
                        project     = ii.project,         # sub project in team
                        dir         = ii.exp_path,
                        config      = dict_to_wandb(ii.d, ignore=ii._wandb_ignore),
                        mode        = wb_mode,
                        id          = ii.exp_id,
                        settings    = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
                )
                
        else:
            ii.submit = False
            
            print('Server -> hostname', ii.server, ii.hostname)
            # if not re.match(ii.server, ii.hostname): # if on local, ssh to server and rerun
            #     # sys.exit('submit')
            #     print('Submitting to server \n')
            #     run_cmds([ii._git_commit_cmd, 'git push origin main --force'], cwd=ii.project_dir)
            #     run_cmds_server(ii.server, ii.user, ii._git_pull_cmd, ii.server_project_dir)  
            #     run_cmds_server(ii.server, ii.user, ii._run_single_cmd, ii.run_dir)                
            #     sys.exit(f'Submitted to server \n')
                ##############################################################################

            # run_cmds([ii._git_commit_cmd, 'git push origin main'], cwd=ii.project_dir)
            
            if ii.run_sweep:
                ii.sweep.parameters |= dict((k, dict(value=v)) for k,v in ii._not_in_sweep.items())
                
                pprint.pprint(ii.sweep.parameters)
                
                ii.sweep_id = wandb.sweep(
                    sweep   = ii.sweep.d, 
                    entity  = ii.wandb_c.entity,
                    project = ii.project,
                )
                
            n_sweep = [len(v['values']) for k,v in ii.sweep.parameters.items() if 'values' in v] 
            n_job = reduce(lambda a,b: a*b, n_sweep if ii.run_sweep else [1,])
            print(f'Running {n_job} on slurm')
            if 0 < cap:
                ii.log(dict(slurm_init=dict(sbatch=ii.sbatch)), create=True, log_name='slurm_init.log')
                for sub in range(1, n_job+1):
                    ii.exp_id = gen_alphanum(n=7)
                    Slurm(**ii.slurm.d).sbatch(ii.sbatch)
            folder = f'runs/{ii.exp_id}' if not ii.run_sweep else f'sweeps/{ii.sweep_id}'
            sys.exit(f'https://wandb.ai/{ii.wandb_c.entity}/{ii.project}/{folder}')

    def _convert(ii, device, dtype):
        import torch
        d = get_cls_dict(ii, sub_cls=True, flat=True)
        d = {k:v for k,v in d.items() if isinstance(v, (np.ndarray, np.generic, list))}
        d = {k:torch.tensor(v, dtype=dtype, device=device, requires_grad=False) for k,v in d.items() if not isinstance(v[0], str)}
        ii.merge(d)

    def _outline(ii):
        print(ii.project_dir)

    @property
    def sbatch(ii,):
        s = [f"""\
module purge
source ~/.bashrc
module load foss
module load CUDA/11.7.0
conda activate {ii.env}

echo $SLURM_JOB_GPUS
echo break
echo $SLURM_JOB_NODELIST
NODEFILE=$(scontrol show hostnames | tr '[:space:]' ',')
echo $NODEFILE

set | grep SLURM | while read line; do echo "# $line"; done

nvidia-smi

srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label hostname &
srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label hostname &
wait
"""
]
        # https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/job-submit-commands-examples
        

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# module purge
# source ~/.bashrc
# module load foss
# module load CUDA/11.7.0
# # export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# conda activate lumi
# # 192GB 

# # srun  python run.py --run_name run.py --seed 808017424 --dtype float32 --n_step 2000 --log_metric_step 10 --log_state_step 10 --charge 0 --spin 0 --a [[0.0,0.0,0.0]] --a_z [4.0] --n_b 256 --n_corr 50 --n_equil 10000 --acc_target 0.5 --with_sign False --n_sv 32 --n_pv 16 --n_fb 2 --n_det 1 --terms_s_emb ['ra','ra_len'] --terms_p_emb ['rr','rr_len'] --ke_method vjp --job_type training --mail_type FAIL --partition sm3090 --nodes 1-1 --cpus_per_task 8 --time 0-01:00:00 --TMP dump/tmp --exp_id Wvyonoa --run_sweep False --user amawi --server svol.fysik.dtu.dk --git_remote origin --git_branch main --env lumi --debug False --wb_mode online --submit False --cap 3 --exp_path dump/exp/demo-12/Wvyonoa --exp_name demo --n_gpu 2 --device_type cuda --head True & 
# # srun --gres=gpu:RTX3090:1 --ntasks=1 --label --exact python run.py --run_name run.py --seed 808017424 --dtype float32 --n_step 2000 --log_metric_step 10 --log_state_step 10 --charge 0 --spin 0 --a [[0.0,0.0,0.0]] --a_z [4.0] --n_b 256 --n_corr 50 --n_equil 10000 --acc_target 0.5 --with_sign False --n_sv 32 --n_pv 16 --n_fb 2 --n_det 1 --terms_s_emb ['ra','ra_len'] --terms_p_emb ['rr','rr_len'] --ke_method vjp --job_type training --mail_type FAIL --partition sm3090 --nodes 1-1 --cpus_per_task 8 --time 0-01:00:00 --TMP dump/tmp --exp_id Wvyonoa --run_sweep False --user amawi --server svol.fysik.dtu.dk --git_remote origin --git_branch main --env lumi --debug False --wb_mode online --submit False --cap 3 --exp_path dump/exp/demo-12/Wvyonoa --exp_name demo --n_gpu 2 --device_type cuda --head False & 
# # wait 
        
#         module load foss
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

        

        # submit_cmd = dict(
        #     cuda=f'CUDA_VISIBLE_DEVICES=""',
        #     rocm=f''
        # )[ii.device_type]

        submit_cmd = dict(
            sweep=f'wandb agent {ii.sweep_path_id}',
            python=f'python {ii.run_name} {ii.cmd}'
        )[ii.exe_mode]
        if ii.distribute:
            # https://groups.google.com/g/slurm-users/c/VpdG0IFZ4n4
            for i in range(ii.n_gpu):
                
                ii.seed = np.random.randint(1, 10000000000)
                
                submit_cmd = dict(
                    sweep=f'wandb agent {ii.sweep_path_id}',
                    python=f'python {ii.run_name} {ii.cmd}'
                )[ii.exe_mode]
        
                device_log_path = ii.exp_path/'slurm'/(str(i)+"_device.log")
                head = [True, False][min(i, 1)]
                
                s += [f'srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label \
                      {submit_cmd} --head {head} > {device_log_path} 2>&1 & \n' ]
            s += ['wait \n']
        else:
            s += [submit_cmd + ' --head True']
        
        return '\n'.join(s)

    @property
    def cmd(ii):
        cmd_d = get_cls_dict(ii, sub_cls=True, ignore=ii._ignore + ['sweep', 'head'], to_cmd=True, flat=True)
        return ' '.join([f'--{k} {v}' for k,v in cmd_d.items() if v])
        
    @property
    def d(ii):
        return get_cls_dict(ii, sub_cls=True, prop=True, ignore=ii._ignore)
    
    def partial(ii, f:Callable, d=None, get_d=False, print_d=False, **kw):
        d = flat_any(d if d else ii.d)
        d_k = inspect.signature(f.__init__).parameters.keys()
        d = {k:copy(v) for k,v in d.items() if k in d_k} | kw
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
                    v = type_me(v, ref[k])
                    try:
                        setattr(cls, k, copy(v))
                        assigned = True
                    except Exception:
                        print(f'Unmerged {k} at setattr')
            if not assigned:
                print(k, v, 'not assigned')
                
    
    def accumulate(ii, step: int, v_tr:dict):
        
        v_path_fn = lambda k, step, dist_id: ii.dist.exchange_dir / f'{k}_{step}_{dist_id}'
        
        for k,v in v_tr.items():
            np.save(v_path_fn(k, step, ii.dist.dist_id), v)

        if ii.head:
            for k,v in v_tr.items():
                # v_path_all = [v_path_fn(k, step, dist_id) for dist_id in ii.dist.dist_id_all]
                
                while not len(ii.dist.exchange_dir.glob(f'{k}_')) == ii.n_gpu:
                    sleep(0.01)

                v_path_all = ii.dist.exchange_dir.glob(f'{k}_')
                tree_flat_all = None
                for p in v_path_all:
                    tree = np.load(p)
                    
                    tree_flat, tree_spec = optree.tree_flatten(tree)
                    if tree_flat_all is None:
                        tree_flat_all = [v[None] for v in tree_flat]
                    else:
                        tree_flat_all = [np.concatenate([v_all, v[None]], axis=0) for v_all, v in zip(tree_flat_all, tree_flat)]
               
                [p.unlink() for p in v_path_all]
                
                tree_flat_mean = [v.mean(0) for v in tree_flat_all]
                tree_mean = optree.tree_unflatten(tree_spec, tree_flat_mean)
                
                [np.save(p/'-mean', tree_mean) for p in v_path_all]
        
        
        for k,v in v_tr.items():
            
            if not k in ii.dist.sync if ii.dist.sync else True:
                _ = v_tr.pop(k)
                continue
            
            v_path = v_path_fn(k, step, ii.dist.dist_id) / '.npy'

            while not v_path.exists():
                sleep(0.01)
                
            v = np.load(p)
            v_tr[k] = v
            p.unlink()
            
        return v_tr
    
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
        
    def log(ii, info: Union[dict,str], create=False, log_name='dump/log.tmp'):
        mkdir(ii.exp_path)
        mode = 'w' if create else 'a'
        info = pprint.pformat(info)
        for p in [log_name, ii.exp_path/log_name]:
            with open(p, mode) as f:
                f.writelines(info)

def get_cls_dict(
        cls,
        ref:Union[list, dict]=None,
        sub_cls=False, 
        fn=False, 
        prop=False, 
        hidn=False,
        ignore:list=None,
        add:list=None,
        to_cmd:bool=False,
        flat:bool=False
    ) -> dict:
    # ref > ignore > add
        ignore = cls._ignore + (ignore or [])
        
        items = []
        for k, v_cls in cls.__class__.__dict__.items():
            
            keep = k in ref if ref else True
            keep = False if (k in ignore) else keep
            keep |= k in add if add else False
            
            if keep:
                if not ref:
                    if k.startswith('__'):
                        continue    
                    if (not hidn) and k.startswith('_'):
                        continue
                    if (not fn) and isinstance(v_cls, partial):
                        continue
                    if (not prop) and isinstance(v_cls, property):
                        continue
    
                v = getattr(cls, k)
                
                if sub_cls and isinstance(v, Sub) or (k in ref if ref else False):
                    v = get_cls_dict(
                        v, ref=ref, sub_cls=False, fn=fn, prop=prop, hidn=hidn, ignore=ignore, to_cmd=to_cmd, add=add)
                    if flat:
                        items.extend(v.items())
                        continue
                    
                items.append([k, v])     
        
        if to_cmd:
            items = ((k, (v.tolist() if isinstance(v, np.ndarray) else v)) for (k,v) in items)
            items = ((str(k).replace(" ", ""), str(v).replace(" ", "")) for (k,v) in items)
                  
        return dict(items)



# def accumulate(ii, step: int, v_tr:dict):
        
#         v_path_fn = lambda k, step, dist_id: ii.dist.exchange_dir / f'{k}_{step}_{dist_id}'
        
#         for k,v in v_tr.items():
#             np.save(v_path_fn(k, step, ii.dist.dist_id), v)

#         if ii.head:
#             for k,v in v_tr.items():
#                 v_path_all = [v_path_fn(k, step, dist_id) for dist_id in ii.dist.dist_id_all]
                
#                 while not all([p.with_suffix('.npy') for p in v_path_all]):
#                     sleep(0.01)

#                 tree_flat_all = None
#                 for p in v_path_all:
#                     tree = np.load(p)
                    
#                     tree_flat, tree_spec = optree.tree_flatten(tree)
#                     if tree_flat_all is None:
#                         tree_flat_all = [v[None] for v in tree_flat]
#                     else:
#                         tree_flat_all = [np.concatenate([v_all, v[None]], axis=0) for v_all, v in zip(tree_flat_all, tree_flat)]
               
#                 [p.unlink() for p in v_path_all]
                
#                 tree_flat_mean = [v.mean(0) for v in tree_flat_all]
#                 tree_mean = optree.tree_unflatten(tree_spec, tree_flat_mean)
                
#                 [np.save(p/'-mean', tree_mean) for p in v_path_all]
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

@property
    def wandb_cmd(ii):
        d = flat_dict(get_cls_dict(ii, sub_cls=True, ignore=['sweep',] + list(ii.sweep.parameters.keys()), add=['exp_path',]))
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
        cmd_d = {str(k).replace(" ", ""): str(v).replace(" ", "") for k,v in d.items()}
        cmd = ' '.join([f' --{k}={v}' for k,v in cmd_d.items() if v])
        return cmd



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
    
    
"""