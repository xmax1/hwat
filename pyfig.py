from pathlib import Path
import paramiko
import sys
import subprocess
from walle.bureaucrat import iterate_folder, mkdir, gen_alphanum
from walle.idiomatic import zip_in_n_chunks, flat_list
from simple_slurm import Slurm
from pprint import pprint
from functools import reduce
import wandb
from time import sleep

this_dir = Path(__file__).parent
TMP = mkdir('./tmp/out')

class _Sub:
    
    def __init__(_i, parent) -> None:
        super().__init__()
        _i.parent = parent
    
    @property
    def dict(_i,):
        d = _i._dict_from_cls(_i,)
        for k,v in d.items():
            if issubclass(type(v), _Sub):  # type(v) on an instantiated class returns the class, which is a subclass of _Sub
                d[k] = _i._dict_from_cls(v)
        return d

    @staticmethod
    def _dict_from_cls(cls):
        return {k: getattr(cls, k) for k in dir(cls) if not k.startswith('_') and not k in ['dict', 'parent', 'sys_arg', 'cmd']}


# submite the hob
""" pseudo sweep
- submit (local) sweep=True/False _remote=False by default x
- git commit (local) x
- check how many jobs are running, if above cap stop and blare alarms x
- git checkout (remote) 
- goes to cluster, runs same file, gets wandb agent config, **** _remote set*** 

"""

"""
LOCAL
- add, commite, push

SERVER
- check #, pull 
"""



""" TODO
- wandb project setup
- submission test 
- copy notes to new project
- simple cmd input 
- get number of jobs in the queue and make sure not overrunning
- SIMPLIFY DIFF MODEL
- calling iterate twice!
"""
""" DOCS

## These things are nice
Minimal code 
    - (only thing you need to know is property, nothing to check / learn / understand about how to use)
Nested class layout 
    - (clarity on subvariables / grouping)
Dot notation
    - c.hypam

## Actually useful
Cascading variables 
    - (set dependencies on the Pyfig root class)
Pure python 
    - (any types you want)
Triple threat class (dict, class, cmd)
    - pyfig/subpyfig -> dict (easy)
        - **c.dict notation for clean, low error, and easy variable submission
    - pyfig -> cmd (nice)
    - dict -> cmd
    - cmd -> dict
    - dict -> pyfig
    - cmd -> pyfig
Cmd line input args
    - Automated capture (no more argparser etc), 
    - flexible input (can just add new vars and type is guessed), 
    - completely typed (anything you want)
    - (cool)
Git version control
    - Commits on run, dumps pyfig to cmd.txt
Directory structure
    - Consistent, clean, well documented structure and creation managed here
Flexib

## kinky
- subclasses are flagged as not initted but they are inited in init

## Cmd
--flag_true0 -flag arg0 --flag arg1 --flag_true1

## Design decisions
- Dump to a pyfig? 
- Regex cmd line parsing?
- loss -> loss table load?
- metric -> metric table load? 
- redunancy in pyfig strings for getting tables? 
- no two parameters can have the same name
redundancies
"""

class Pyfig(_Sub):

    project_dir:        Path    = Path().absolute()
    n_device:           int     = property(lambda _: count_gpu())
    
    seed:               int     = 808017424         # grr

    exp_name:           str     = 'junk'
    run_path:           Path    = property(lambda _: _.project_dir / 'run.py')
    data_dir:          Path     = Path().home() / 'data'
    
    loss_type = 'l1'
    half_precision = True
    dtype:              str     = 'f32'
    n_step:             int     = 1000

    n_layer:            int     = 3
    af:                 str     = 'tanh'    # activation function
    
    class ema(_Sub):
        beta = 0.995
        update_every = 10
        update_after_step = 100
        inv_gamma = 1.0
        power = 2 / 3.
        min_value = 0.0

    class ddpm(_Sub):
        beta_schedule = 'linear'
        timesteps = 1000
        p2_loss_weight_gamma = 0. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1
        _i_condition = False # not tested yet
        is_pred_x0 = False # by default, the model will predict noise, if True predict x0
        update_after_step = 1
        update_every = 1

    class data(_Sub):
        dataset = 'fashion_mnist'
        b_size = 16
        cache = False
        image_size = 28
        channels = 1

    class model(_Sub):
        dim = 64
        dim_mults = (1, 2, 4)

    class opt(_Sub):
        optimizer = 'Adam'
        beta1 = 0.9
        beta2 = 0.99
        eps = 1e-8
        lr = 0.001
        loss_fn = 'l1_loss'  # change this to loss table load? 

    class sweep(_Sub):
        method = 'random'
        name = 'sweep'
        metrics = dict(
            goal = 'minimize',
            name = 'validation_loss',
        )
        parameters = dict(
            batch_size = {'values': [16, 32, 64]},
            epoch = {'values': [5, 10, 15]},
            lr = {'max': 0.1, 'min': 0.0001},
        )
        n_sweep = reduce(lambda i0,i1: i0*i1, [len(v['values']) for k,v in parameters.items() if 'values' in v])
        sweep_id = ''

    class wandb(_Sub):
        job_type:           str     = 'training'
        entity:             str     = 'xmax1'

    log_sample_step:    int     = 1000
    log_state_step:     int     = 1000         # wandb entity
    log_metric_step:    int     = 100
    n_epoch:            int     = 10

    class slurm(_Sub):
        output          = TMP / 'o-%j.out'
        error           = TMP / 'e-%j.err'
        mail_type       = 'FAIL'
        partition       ='sm3090'
        nodes           = 1                # n_node
        ntasks          = 8                # n_cpu
        cpus_per_task   = 1     
        time            = '0-12:00:00'     # D-HH:MM:SS
        gres            = 'gpu:RTX3090:1'
        job_name        = property(lambda sub_i: sub_i.parent.exp_name)  # this does not call the instance it is in
        sbatch          = property(lambda sub_i: 
            f""" 
            module purge 
            source ~/.bashrc 
            module load GCC 
            module load CUDA/11.4.1 
            module load cuDNN/8.2.2.26-CUDA-11.4.1 
            conda activate {sub_i.parent.env} 
            export MKL_NUM_THREADS=1 
            export NUMEXPR_NUM_THREADS=1 
            export OMP_NUM_THREADS=1 
            export OPENBLAS_NUM_THREADS=1
            pwd
            nvidia-smi
            mv_cmd = f'mv {TMP}/o-$SLURM_JOB_ID.out {TMP}/e-$SLURM_JOB_ID.err $out_dir' 
            """
        )

    exp_id:             str     = gen_alphanum(n=7)

    project:            str     = 'diff_model'
    
    server_project_dir: Path    = property(lambda _i: _i.project_dir)
    project_exp_dir:    Path    = property(lambda _i: _i.project_dir / 'exp')
    project_cfg_dir:    Path    = property(lambda _i: _i.project_dir / 'cfg')
    iter_exp_dir:       bool    = True
    exp_path:           Path    = property(lambda _i: iterate_folder(_i.project_exp_dir / _i.exp_name, _i.iter_exp_dir) / _i.exp_id)

    server:             str     = 'svol.fysik.dtu.dk'   # SERVER
    user:               str     = 'amawi'     # SERVER
    entity:             str     = 'xmax1'       # WANDB entity
    git_remote:         str     = 'origin'      
    git_branch:         str     = 'main'        

    env:                str     = 'dex'            # CONDA ENV
    commit_id:          str     = None
    
    _n_run_slurm:       int     = None # utility var to turn off submission / avoid recursion on slurm submission

    sys_arg: list = sys.argv[1:]

    def __init__(_i, args: dict={}, cap=40) -> None:
        for k, v in Pyfig.__dict__.items():
            if isinstance(v, type):
                setattr(_i, k,  v(_i))  # initialising subclasses 
        
        _i._update(args | _i._cmd_to_dict(sys.argv[1:]))
        print(f'created exp_dir {_i.exp_path}')
        
        if _i._n_run_slurm:
                       
            n_job_running = _i.run_cmds([f'squeue -u {_i.parent.user} -h -t pending,running -r | wc -l'])
            if n_job_running > cap:
                raise exit(f'There are {n_job_running} on the cluster cap is {cap}')
                
            _slurm = Slurm(
                output          = TMP / 'o-%j.out',
                error           = TMP / 'e-%j.err',
                mail_type       = _i.slurm.mail_type,
                partition       = _i.slurm.partition,
                nodes           = _i.slurm.nodes,              
                ntasks          = _i.slurm.ntasks,         
                cpus_per_task   = _i.slurm.cpus_per_task,     
                time            = _i.slurm.time,   
                gres            = _i.slurm.gres,
                job_name        = _i.slurm.job_name,
            )

            n_run, _i._n_run_slurm = _i._n_run_slurm, 0  # This is the magic. _run_slurm makes the winding path work.

            for _ in range(n_run):
                _slurm.sbatch(_i.slurm.sbatch 
                + f'out_dir={(mkdir(_i.exp_path/"out"))} {_i.cmd} | tee $out_dir/py.out date "+%B %V %T.%3N" ')

    @property
    def cmd(_i,):
        d = _i._dict_from_cls(_i,) # is flat
        return _i._dict_to_cmd(d)

    @staticmethod
    def _dict_to_cmd(d:dict):
        return ' '.join([f' --{k}  {str(v)} ' for k,v in d.items()])

    def _cmd_to_dict(_i, cmd, _d={}):
        booleans = ['True', 'true', 't', 'False', 'false', 'f']
        """
        fmt: [--flag, arg, --true_flag, --flag, arg1]
        # all flags double dash because of negative numbers duh """
        delim = ' --' 
        cmd = [x.lstrip().lstrip('--').rstrip() for x in cmd.split(delim)]
        cmd = [x.split(' ', maxsplit=1) for x in cmd if ' ' in x]
        [x.append('True') for x in cmd if len(x) == 1]
        cmd = flat_list(cmd)
        cmd = iter([x.strip() for x in cmd])

        _default = _i.dict
        for k,v in zip(cmd, cmd):
            if v in booleans:
                v = booleans.index(v) < 3  # 0-2 True 3-5 False
            if k in _default:
                _d[k] = type(_default[k])(v)
            else:
                from ast import literal_eval
                try:
                    _d[k] = literal_eval(v)
                except:
                    _d[k] = v
                print(f'Guessing type: {k} as {type(v)}')
        return _d

    def flat_dict(_i, d:dict, items:list=[]):
        for k,v in d.items():
            if isinstance(v, dict):
                items.extend(_i.flat_dict(v).items())
            else:
                items.append((k, v))
        return dict(items)

    def _update(_i, d:dict):
        for k,v in d.items():
            updated = _i._update_recurse(_i, k, v)
            if updated:
                continue
            else:
                print('Not updated: ', k, v)

    def _update_recurse(_i, cls, k, v):
        d: dict = cls.dict
        if k in d:
            cls.__dict__[k] = type(d[k])(v)
            return True
        for cls_v in d.values():
            if issubclass(type(cls_v), _Sub):
                if _i._update_recurse(cls_v, k, v):
                    return True
        return False

    def run_cmds(cmd:str|list, cwd:str|Path=None, input_req:str=None)->str:
        if isinstance(cmd, str):
            cmd = [cmd]
        for cmd_1 in cmd:
            stdout = subprocess.run([c.strip() for c in cmd_1.split(' ')], cwd=cwd, input=input_req, capture_output=True)
        return stdout

    def run_cmds_server(_i, cmd:str|list, cwd=None):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # if not known host
        client.connect(hostname=_i.parent.server, username=_i.parent.user)
        if isinstance(cmd, str):
            cmd = [cmd]
        with client as _r:
            for cmd_1 in cmd:
                _r.exec_command(f'cd {cwd}; {cmd_1}')[1] # in, out, err
                sleep(0.5)

    def submit_to_cluster(_i, sweep=False, msg=None, cap=40):
        if not (_i._n_run_slurm is None):
            _i._n_run_slurm = 1

            msg = msg or _i.exp_id

            _i.run_cmds(['git add .', f'git commit -m {msg}', 'git push'], cwd=_i.project_dir)
            stdout = _i.run_cmds(['git log -l'], cwd=_i.project_dir).decode('utf-8')
            _i.commit_id = stdout.replace('\n', ' ').split(' ')[1]

            if sweep:
                _i.sweep_id = wandb.sweep(
                    env     = f'conda activate {_i.env};',
                    sweep   = _i.sweep, 
                    program = _i.run_path,
                    project = _i.project,
                    name    = _i.exp_name,
                    run_cap = _i.sweep.n_sweep
                )
                _i._n_run_slurm *= _i.sweep.n_sweep

            _i.run_cmds_server([f'python -u {_i.run_path} '+_i.cmd,])


# def _touch(_i, d:dict={}):
#         _default = _i.flat_dict(_i.dict)
#         d = _i.flat_dict(d)
#         for k,v in _default.items():
#             if k in d:
#                 del d[k]
#                 updated = _i._touch_recurse(_i, k, v)
#             else:
#                 d[k] = v
#             if updated:
#                 continue
#             else:
#                 print('Not updated: ', k, v)
#         return d

#     def _touch_recurse(_i, cls, k, v):
#         d: dict = cls.dict
#         if k in d:
#             cls.__dict__[k] = type(d[k])(v)
#             return True
#         for cls_v in d.values():
#             if issubclass(type(cls_v), _Sub):
#                 if _i._touch_recurse(cls_v, k, v):
#                     return True
#         return False

    # def _type_sys_new():
    #     from ast import literal_eval
    #     out_arg = {}
    #     for k, v in {k:v for k,v in zip_in_n_chunks(arg, 2)}.items():
    #         if v in booleans:  # ADDING REDUNDANCY: For if the boolean argument is mispelt
    #             out_arg[k] = ('t' in v) or ('T' in v)
    #         else:
    #             try:
    #                 out_arg[k] = literal_eval(v)
    #             except:
    #                 out_arg[k] = v  # strings don't work in literal eval mk YYYY
    #     return out_arg

    # def _guess_type():
    #     flag_var_bools = ['True', 'true', 't', 'False', 'false', 'f']

    




### GPUS ###
def get_env_var(k: str):
    return run_cmd(f'echo ${k}') 

def set_env_var(k: str):
    run_cmd(f'export ${k}')
    return 

def count_gpu() -> int:
    # output = run_cmd('echo $CUDA_VISIBLE_DEVICES', cwd='.')
    import os
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    cvd = None if cvd == '' else cvd
    return len(cvd.split(',')) if not cvd is None else 0 

def get_gpu_utilisation() -> dict:
    """ Find all GPUs and their total memory """
    output = run_cmd('nvidia-smi --query-gpu=index,memory.total --format=csv')
    total_memory = dict(row.replace(',', ' ').split()[:2] for row in output.strip().split('\n')[1:])
    return {gpu_id: {'used': 0, 'used_by_others': 0, 'total': int(total)} for gpu_id, total in total_memory.items()}

def get_free_gpu_id() -> int | None:
    ut_all = get_gpu_utilisation()
    for idx, ut in enumerate(ut_all):
        if ut < 100:
            return idx
    return None

