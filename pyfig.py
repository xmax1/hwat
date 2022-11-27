
from hwat import compute_r, compute_rvec, compute_emb

from utils import * # bad practice - no you're bad practise :mic drop:

class Pyfig(Sub):

    seed:               int     = 808017424 # grr
    project_root:       str     = Path().home()/'projects'

    project:            str     = 'hwat'
    project_path:       Path    = property(lambda _: _.project_root / _.project)
    server_project_path:Path    = property(lambda _: _.project_path)
    n_device:           int     = property(lambda _: count_gpu())

    exp_name:           str     = 'junk'
    run_path:           Path    = property(lambda _: _.project_path / 'run.py')
    data_dir:           Path    = project_root / 'data'
    
    half_precision:     bool    = True
    dtype:              str     = 'f32'
    n_step:             int     = 1000

    af:                 str     = 'tanh' # activation function
    n_layer:            int     = 3
    
    class data(Sub):
        b_size: int  = 16
        n_e: int
        n_u: int
        n_d: int = property(lambda _: _.n_e-_.n_u)
        cache      = False

    class model(Sub):
        n_sv: int       = 16
        n_pv: int       = 8
        n_fb: int       = 2
        n_fb_out: int   = property(lambda _: _.n_sv*3+_.n_pv*2)
        n_det: int      = 1

        class af:
            tanh = nn.tanh

        af_fb: Callable     = af.tanh
        af_fb_out: Callable = af.tanh
        af_pseudo: Callable = af.tanh

        terms_s_emb = ['x_rlen']
        terms_p_emb = ['xx']

        def compute_emb(x, *, a=None, terms=[]):
            z = []
            if 'x' in terms:
                z += [x]
            if 'x_rlen' in terms:
                z += [jnp.linalg.norm(x, axis=-1, keepdims=True)]
            if 'xa' in terms:
                z += [compute_rvec(x, a)]
            if 'xa_rlen' in terms:
                z += [compute_r(x, a)]
            if 'xx' in terms:
                z += [compute_r(x, x)]
            if 'xx_rlen' in terms:
                z += [compute_rvec(x, x)]
            return jnp.concatenate(z, axis=-1)

        compute_s_emb: Callable = \
            property(lambda _: partial(compute_emb(terms=_.terms_s_emb)))
        compute_p_emb: Callable = \
            property(lambda _: partial(compute_emb(terms=_.terms_p_emb)))
        
    class opt(Sub):
        optimizer   = 'Adam'
        beta1       = 0.9
        beta2       = 0.99
        eps         = 1e-8
        lr          = 0.001
        loss        = 'l1'  # change this to loss table load? 

    class sweep(Sub):
        method      = 'random'
        name        = 'sweep'
        metrics = dict(
            goal    = 'minimize',
            name    = 'validation_loss',
        )
        parameters = dict(
            batch_size  = {'values' : [16, 32, 64]},
            epoch       = {'values' : [5, 10, 15]},
            lr          = {'max'    : 0.1, 'min': 0.0001},
        )
        n_sweep = run_cap = reduce(
            lambda i,j:i*j,[len(v['values']) for k,v in parameters.items() if 'values' in v])+1
        sweep_id = ''

    class wandb(Sub):
        job_type:       str     = 'training'
        entity:         str     = 'xmax1'

    log_sample_step:    int     = 5
    log_metric_step:    int     = 5
    log_state_step:     int     = 10         # wandb entity
    n_epoch:            int     = 20

    class slurm(Sub):
        
        mail_type       = 'FAIL'
        partition       ='sm3090'
        nodes           = 1                # n_node
        ntasks          = 8                # n_cpu
        cpus_per_task   = 1     
        time            = '0-12:00:00'     # D-HH:MM:SS
        gres            = 'gpu:RTX3090:1'
        output          = property(lambda _: _.parent.TMP)/'o-%j.out'
        error           = property(lambda _: _.parent.TMP)/'e-%j.err'
        job_name        = property(lambda _: _.parent.exp_name)  # this does not call the instance it is in
        sbatch          = property(lambda _: f""" 
            module purge 
            source ~/.bashrc 
            module load GCC 
            module load CUDA/11.4.1 
            module load cuDNN/8.2.2.26-CUDA-11.4.1 
            conda activate {_.parent.env} 
            export MKL_NUM_THREADS=1 
            export NUMEXPR_NUM_THREADS=1 
            export OMP_NUM_THREADS=1 
            export OPENBLAS_NUM_THREADS=1
            pwd
            nvidia-smi
            mv_cmd = f'mv {_.parent.TMP}/o-$SLURM_JOB_ID.out {_.parent.TMP}/e-$SLURM_JOB_ID.err $out_dir' 
    """)

    exp_id:             str     = gen_alphanum(n=7)
    
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
    commit_id:          str     = property(lambda _: _.get_commit_id())
    
    _sys_arg: list = sys.argv[1:]
    _submit_state: int = -1

    def __init__(_i,args:dict={},cap=40,wandb_mode='online',notebook=False):
        super().__init__()
        _i.__safe_init__()

        update_cls_with_dict(_i,args)
        if not notebook:
            update_cls_with_dict(cmd_to_dict(sys.argv[1:],_i.dict))

        wandb.init(
            job_type    = _i.wandb.job_type,
            entity      = _i.wandb.entity,
            project     = _i.project,
            dir         = _i.exp_path,
            config      = dict_to_wandb(_i.dict),
            mode        = wandb_mode,
            settings=wandb.Settings(start_method='fork'), # idk y this is issue, don't change
        )

        if _i._submit_state > 0:
            n_job_running = run_cmds([f'squeue -u {_i.user} -h -t pending,running -r | wc -l'])
            if n_job_running > cap:
                exit(f'There are {n_job_running} on the submit cap is {cap}')

            _slurm = Slurm(**_i.slurm.dict)

            n_run, _i._submit_state = _i._submit_state, 0            
            for _ in range(n_run):
                _slurm.sbatch(_i.slurm.sbatch 
                + f'out_dir={(mkdir(_i.exp_path/"out"))} {_i.cmd} | tee $out_dir/py.out date "+%B %V %T.%3N" ')

    @property
    def cmd(_i,):
        d = flat_dict(_i.dict)
        return ' '.join([f' --{k}  {str(v)} ' for k,v in d.items()])

    @property
    def commit_id(_i,)->str:
        process = run_cmds(['git log --pretty=format:%h -n 1'], cwd=_i.project_path)[0]
        return process.stdout.decode('utf-8') 

    def submit(_i, sweep=False, commit_msg=None, commit_id=None):
        commit_msg = commit_msg or _i.exp_id
        _i._submit_state *= -1
        if _i._submit_state > 0:
            if sweep:
                _i.sweep_id = wandb.sweep(
                    env     = f'conda activate {_i.env};',
                    sweep   = _i.sweep.dict, 
                    program = _i.run_path,
                    project = _i.project,
                    name    = _i.exp_name,
                    run_cap = _i.sweep.n_sweep
                )
                _i._submit_state *= _i.sweep.n_sweep
            
            local_out = run_cmds(['git add .', f'git commit -m {commit_msg}', 'git push'], cwd=_i.project_path)

            cmd = f'python -u {_i.run_path} ' + (commit_id or _i.commit_id) + _i.cmd
            server_out = run_cmds_server(_i.server, _i.user, cmd, cwd=_i.server_project_path)

