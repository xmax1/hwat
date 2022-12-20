# TORCH MNIST DISTRIBUTED EXAMPLE

"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pyfig import Pyfig
import numpy as np

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        
""" Distributed Synchronous SGD Example """
def run(c: Pyfig):
    torch.manual_seed(1234)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    n_device = c.n_device
    print(f'🤖 {n_device} GPUs available')

    ### model (aka TrainState) ### 
    from functools import partial
    import jax
    import optax
    from flax.training.train_state import TrainState
    from hwat import FermiNet

    @partial(jax.pmap, axis_name='dev', in_axes=(0,0))
    def create_train_state(rng, r):
        model = c.partial(FermiNet)
        params = model.init(rng, r)
        opt = optax.chain(optax.clip_by_block_rms(1.),optax.adamw(0.001))
        return TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    ### train step ###
    from jax import numpy as jnp
    from hwat import compute_ke_b, compute_pe_b
    from typing import NamedTuple

    @partial(jax.pmap, in_axes=(0, 0))
    def train_step(state, r_step):

        ke = compute_ke_b(state, r_step)
        pe = compute_pe_b(r_step, c.data.a, c.data.a_z)
        e = pe + ke
        
        e_mean_dist = jnp.mean(jnp.abs(jnp.median(e) - e))
        e_clip = jnp.clip(e, a_min=e-5*e_mean_dist, a_max=e+5*e_mean_dist)

        def loss(params):
            return ((e_clip - e_clip.mean())*state.apply_fn(params, r_step)).mean()
        
        grads = jax.grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        
        v_tr = dict(
            params=state.params, grads=grads,
            e=e, pe=pe, ke=ke,
            r=r_step
        )

        return state, v_tr


    ### init variables ###
    from utils import gen_rng
    from hwat import init_r, get_center_points
    from jax import random as rnd

    rng, rng_p = gen_rng(rnd.PRNGKey(c.seed), c.n_device)
    center_points = get_center_points(c.data.n_e, c.data.a)
    r = init_r(rng_p, c.data.n_b, c.data.n_e, center_points, std=0.1)
    deltar = jnp.array([0.02])[None, :].repeat(n_device, axis=0)

    print(f"""exp/actual | 
        rng    : {(2,)}/{rng.shape} 
        rng_p  : {(c.n_device,2)}/{rng_p.shape} 
        cps    : {(c.data.n_e,3)}/{center_points.shape}
        r      : {(c.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
        deltar : {(c.n_device, 1)}/{deltar.shape}
    """)


    ### init functions ### 
    from hwat import sample_b

    state = create_train_state(rng_p, r)
    metro_hast = jax.pmap(partial(sample_b, n_corr=c.data.n_corr), in_axes=(0,0,0,0))


    ### train ###
    import wandb
    from hwat import keep_around_points
    from utils import compute_metrix

    wandb.define_metric("*", step_metric="tr/step")
    for step in range(1, c.n_step+1):
        rng, rng_p = gen_rng(rng, c.n_device)

        r, acc, deltar = metro_hast(rng_p, state, r, deltar)
        r = keep_around_points(r, center_points, l=2.) if step < 1000 else r
        
        state, v_tr = train_step(state, r)

        if not (step % c.log_metric_step):
            metrix = compute_metrix(v_tr)
            wandb.log({'tr/step':step, **metrix})
            
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)





if __name__ == "__main__":
    
    ### pyfig ###
    arg = dict(
        charge = 0,
        spin  = 0,
        a = np.array([[0.0, 0.0, 0.0],]),
        a_z  = np.array([4.,]),
        n_b = 512, 
        n_sv = 32, 
        n_pv = 32, 
        n_corr = 20, 
        n_step = 10000, 
        log_metric_step = 50, 
        exp_name = 'demo',
        # sweep = {},
    )
    
    c = Pyfig(wb_mode='online', arg=arg, submit=True, run_sweep=True)

    
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
         
    
    
    """ live plotting in another notebook """
    """ copy lines and run in analysis while the exp is live """
    # api = wandb.Api()
    # run = api.run("<run-here>")
    # c = run.config
    # h = run.history()
    # s = run.summary
    
    # ### fancy logging variables, philosophically reminding us of the goal ###
    # fancy = dict(
    #         pe		= r'$V(X)',    				
    #         ke		= r'$\nabla^2',    		
    #         e		= r'$E',						
    #         log_psi	= r'$\log\psi', 			
    #         deltar	= r'$\delta_\mathrm{r}',	
    #         x		= r'$r_\mathrm{e}',
    # )

