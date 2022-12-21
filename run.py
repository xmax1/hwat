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
    
    torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
    
    n_device = c.n_device
    print(f'🤖 {n_device} GPUs available')

    ### model (aka Trainmodel) ### 
    from functools import partial
    from hwat import Ansatz_fb
    from torch import nn
    model = c.partial(Ansatz_fb)
    
    model: nn.Module
    
    ### train step ###
    from hwat import compute_ke_b, compute_pe_b
    
    def train_step(model, r):

        with torch.no_grad():
            ke = compute_ke_b(model, r)
            pe = compute_pe_b(r, c.data.a, c.data.a_z)
            e = pe + ke
            e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
            e_clip = torch.clip(e, a_min=e-5*e_mean_dist, a_max=e+5*e_mean_dist)

        loss = ((e_clip - e_clip.mean())*model(r)).mean()
        loss.backward()
        
        grads = [p.grad for p in model.parameters()]
        params = [p.item for p in model.parameters()]  # ❗
        
        v_tr = dict(
            params=params, grads=grads,
            e=e, pe=pe, ke=ke,
            r=r
        )
        return v_tr


    ### init variables ###
    from hwat import init_r, get_center_points

    center_points = get_center_points(c.data.n_e, c.data.a)
    r = init_r(n_device, c.data.n_b, c.data.n_e, center_points, std=0.1)
    deltar = torch.tensor([0.02])[None, :]

    print(f"""exp/actual | 
        cps    : {(c.data.n_e,3)}/{center_points.shape}
        r      : {(c.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
        deltar : {(c.n_device, 1)}/{deltar.shape}
    """)


    ### init functions ### 
    from hwat import sample_b

    ### train ###
    import wandb
    from hwat import keep_around_points
    from utils import compute_metrix
    
    ### add in optimiser
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    ### fix sampler
    ### fix train step 
    ### metrix conversion

    wandb.define_metric("*", step_metric="tr/step")
    for step in range(1, c.n_step+1):

        r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr)  # ❗needs testing 
        r = keep_around_points(r, center_points, l=2.) if step < 1000 else r
        
        opt.zero_grad()
        v_tr = train_step(model, r)
        opt.step()
        
        if not (step % c.log_metric_step):
            metrix = compute_metrix(v_tr)  # ❗ needs converting to torch, ie tree maps
            wandb.log({'tr/step':step, **metrix})
            
            
    # for epoch in range(10):
    #     epoch_loss = 0.0
    #     for data, target in train_set:
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = F.nll_loss(output, target)
    #         epoch_loss += loss.item()
    #         loss.backward()
    #         average_gradients(model)
    #         optimizer.step()
    #     print('Rank ', dist.get_rank(), ', epoch ',
    #           epoch, ': ', epoch_loss / num_batches)





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
    
    c = Pyfig(wb_mode='online', arg=arg, submit=False, run_sweep=False)

    ### DISTRIBUTED   # ❗# ❗# ❗# ❗ after single gpu demo
    # size = 2
    # processes = []
    # mp.set_start_method("spawn")
    # for rank in range(size):
    #     p = mp.Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
        
        
        
    # fun stuff after 
    
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

