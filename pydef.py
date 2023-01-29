from pathlib import Path
import torch
import numpy as np
from walle.pyfig_utils import PyfigBase
Base: PyfigBase = PyfigBase()

""" PlugIns



"""



cmd: str = """
python run.py 
--time 01:00:00 
--submit 
--multimode train:eval:max_mem:opt_hypam 
--exp_name something~else 
--n_step 40 
--n_pre_step 20 
--dist naive 
--n_gpu 2 
--n_b 128 
--a_z [4]


"""

user: 				str 	= user

project:            str     = 'hwat'
run_name:       	Path	= 'run.py'
load_exp_state:		str		= ''

exp_name:       	str		= '' # default is demo
exp_id: 			str		= ''
group_exp: 			bool	= False

mode: 				str		= ''
multimode: 			str		= 'train:eval'

debug: 				bool    = False
run_sweep:      	bool    = False

seed:           	int   	= 808017424 # grr
dtype:          	str   	= torch.float # torch.float32
cudnn_benchmark: 	bool 	= False

n_step:         	int   	= 1000
n_pre_step:    		int   	= 250

step: 				int 	= None
log_metric_step:	int   	= 50
log_state_step: 	int   	= property(lambda _: _.n_step//10)

n_eval_step:        int   	= 100
opt_obj:			str		= 'e'
eval_keys: 			list 	= ['e',]
log_keys: 			list 	= ['r', 'e', 'pe', 'ke']

system: 	str			= ''

charge:     int         = 0
spin:       int         = 0
a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
a_z:        np.ndarray  = np.array([16.,])

n_b:        int         = 512
n_corr:     int         = 20
acc_target: int         = 0.5

n_equil_step:int        = property(lambda _: 1000//_.n_corr)
n_e:        int         = property(lambda _: int(sum(_.a_z)))
n_u:        int         = property(lambda _: (_.spin + _.n_e)//2)
n_d:        int         = property(lambda _: _.n_e - _.n_u)


class model(PyfigBase.model):
    compile_ts: 	bool	= False
    compile_func:	bool	= False
    optimise_ts:	bool	= False
    optimise_aot:	bool 	= False
    with_sign:      bool    = False
    functional: 	bool	= True

    terms_s_emb:    list    = ['ra', 'ra_len']
    terms_p_emb:    list    = ['rr', 'rr_len']
    ke_method:      str     = 'grad_grad'
    n_sv:           int     = 32
    n_pv:           int     = 32
    n_fb:           int     = 3
    n_det:          int     = 4
    
    n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)

class opt(PyfigBase.opt):
    opt_name: 		str		= 'RAdam' # 'AdaHessian', 'LBFGS', 
    lr:  			float 	= 0.01
    betas:			list	= [0.9, 0.999]
    eps: 			float 	= 1e-4
    weight_decay: 	float 	= 0.0
    hessian_power: 	float 	= 1.0

    class scheduler(PlugIn):
        _prefix: 	str 	= 'sch_'
        sch_default:str 	='OneCycleLR'

        sch_name: 	str		= 'OneCycleLR'
        sch_max_lr:	float 	= 0.01
        sch_epochs: int 	= 1

class sweep(PyfigBase.sweep):
    sweep_method: 	str		= 'grid'

    parameters: 	dict 	= dict(
        # dtype			=	Param(values=[torch.float32, torch.float64], dtype=str), # !! will not work
        # n_b			=	Param(values=[512, 1024, 2048], dtype=int), # 

        # n_sv:           int     = 64
        # n_pv:           int     = 32
        # n_fb:           int     = 3
        # n_det:          int     = 4

        opt_name		=	Param(values=['AdaHessian',  'RAdam'], dtype=str),
        lr				=	Param(domain=(0.0001, 1.), log=True),
        sch_max_lr		=	Param(values=[0.1, 0.01, 0.001], dtype=float),
        weight_decay	= 	Param(domain=[0.0, 1.], dtype=float, condition=['AdaHessian',]),
        hessian_power	= 	Param(values=[0.5, 1.], dtype=float, condition=['AdaHessian',]),

        # n_sv	= 	Param(values=[16, 32, 64], dtype=int),
        # n_pv	= 	Param(values=[16, 32, 64], dtype=int),
        # n_det	= 	Param(values=[1, 2, 4, 8], dtype=int),
        # n_fb	= 	Param(values=[1, 2, 3, 4], dtype=int),
        # n_b		= 	Param(values=[512, 1024, 2048, 4096], dtype=int),

    )

class dist(naive):
    dist_method = 'naive'

class resource(niflheim):
    env: 			str     = 'zen'
    n_gpu: 			int 	= 1

class wb(PyfigBase.wb):
    wb_mode = 'online'

zweep: str = ''

	def __init__(ii, notebook: bool=False, sweep: dict={}, c_init: dict={}, **other_arg) -> None:

		print('pyfig:init')
		super().__init__(notebook=notebook, c_init=c_init, sweep=sweep, **other_arg)

		print('pyfig:init:system')
		ii.update(systems.get(ii.data.system, {}))
  
		ii.run_local_or_submit()

		"""
		1- Fix the batch size for O2_neutral by mem_maxing 5-25 on base model

		2- Base model
		base = dict(
			n_sv=32,
			n_pv=32,
			n_fb=3,
			n_det=4,
		)

		3- 
		"""

		# tag record to be able to clear anything from wandb with no record tag
		# - todo
		# - run scaling exp

		""" dummy to real ones
		python run.py --submit 
		python run.py --time 03:00:00 --submit --multimode train:eval:max_mem:opt_hypam \
		--exp_name ~debug --n_step 40 --n_pre_step 20 --dist naive --n_gpu 2 --n_b 128 --a_z [4]

		python run.py --time 03:00:00 --submit --multimode train-record:eval --system O2_neutral_triplet \
		--exp_name gpuscale~test --n_step 1000 --n_pre_step 100 --dist hf_accelerate --n_gpu 2 --n_b 512

		python run.py --time 03:00:00 --submit --multimode train-record:eval --n_step 10000 --system O2_neutral_triplet --exp_name gpuscale~v0 --n_pre_step 1000 --dist hf_acclerate --zweep n_gpu-2-4-6-8-10
		python run.py --time 03:00:00 --submit --multimode train-record:eval --n_step 10000 --system O2_neutral_triplet --exp_name gpuscale~v0 --n_pre_step 1000 --dist naive --zweep n_gpu-2-4-6-8-10 --n_b 4096

		# opt_hypam system O2
		python run.py --submit --time 00:05:00 --mode opt_hypam  \
		--system O2_neutral_triplet --exp_name ~debug --dist hf_accelerate --n_gpu 2  \
		--n_step 100  --n_pre_step 50  --n_b 512 --sync_step -1 --n_trials 4
		
		python run.py --submit --time 02:00:00 --mode opt_hypam-record --system O2_neutral_triplet \
		--exp_name opt_hypam~O2 --dist naive --n_gpu 4 --n_step 500  --n_pre_step 500  --n_b 1024 
		--sync_step -1 --log_state_step -1 \
		--n_trials 20


		python run.py --submit --time 01:00:00 --mode opt_hypam \
		--system O2_neutral_triplet --exp_name opt_hypam~O2 --dist naive --n_gpu 4 \
		--n_step 500  --n_pre_step 500  --n_b 1024 --sync_step -1 --n_trials 20

		"""


	python run.py --time 03:00:00 --submit --multimode max_mem:opt_hypam --exp_name sweep~memopt --zweep a_z-16-17
	python run.py --time 03:00:00 --submit --multimode max_mem:opt_hypam --exp_name sweep~memopt --zweep "[[16],[17]]"
	python run.py --time 03:00:00 --submit --multimode max_mem:opt_hypam --exp_name sweep~memopt --zweep a_z-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30


		# python run.py --submit --mode train --n_det 1 --n_step 10000 --a_z [4] --exp_name stab~4 --n_pre_step 1000
		# python run.py --submit --mode train --n_det 1 --n_step 10000 --a_z [16] --exp_name stab~16 --n_pre_step 1000
		# python run.py --submit --dist hf_accelerate --n_gpu 2 --exp_name demo~opt_hypam --mode opt_hypam --time 12:00:00 --system O2_neutral_triplet
		# a_z		= 	Param(values=[[i,] for i in range(5, 50)], dtype=int),
		# get_mem_max 15

		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 01:00:00 --mode --max_mem

		# 26 / 1 / 23
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist naive
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist naive --cudnn_benchmark
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist hf_accelerate
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist hf_accelerate --cudnn_benchmark

		# python run.py --submit --cudnn_benchmark --exp_name opt_hypam1~O2 --mode opt_hypam-record --time 12:00:00 --n_step 500 --n_b 512

		# python run.py --submit --a_z [16] --dist naive --cudnn_benchmark --exp_name sweep-n_b --mode max_mem --time 12:00:00
		# python run.py --submit --dist naive --cudnn_benchmark --exp_name sweep-a_z --group_exp --time 00:10:00
		# python run.py --submit --exp_name dist --group_exp --time 01:00:00 --a_z [30] --dist naive --mode max_mem --n_gpu 10
		# python run.py --submit --exp_name dist --group_exp --time 01:00:00 --a_z [30] --dist hf_accelerate --mode max_mem --n_gpu 10
		# for a_z in [[i,] for i in range(10, 60, 2)]:
		# 	run_d = dict(a_z=a_z)

