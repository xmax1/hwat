

# TODO
- sweep / hypam_opt dict

# github

https://github.com/jettify/pytorch-optimizer
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
https://github.com/NVIDIA/apex
https://github.com/jettify/pytorch-optimizer
https://panel.holoviz.org/getting_started/installation.html

LiH	3.015	-8.07050(1)	-8.0687	-8.0697	-8.070696	-7.98737	-8.07054846	99.94(1)
Li2	5.051	-14.99475(1)	-14.9921	-14.9936	-14.99507	-14.87155	-14.995447	99.47(1)
NH3	-	-56.56295(8)	-56.5535	-56.5591	-56.5644	-56.2247	-	99.57(2)
CH4	-	-40.51400(7)	-40.5067	-40.511	-40.515	-40.2171	-	99.66(3)
CO	2.173	-113.3218(1)	-113.3047	-113.3154	-113.3255	-112.7871	-	99.32(3)
N2	2.068	-109.5388(1)	-109.5224	-109.5327	-109.5425	-108.994	-109.542347	99.36(2)
Ethene	-	-78.5844(1)	-78.5733	-78.5812	-78.5888	-78.0705	-	99.16(2)
Methylamine	-	-95.8554(2)	-95.8437	-	-95.8653	-95.2628	-	98.36(3)
Ozone	-	-225.4145(3)	-225.3907	-225.4119	-225.4338	-224.3526	-	98.42(3)
Ethanol	-	-155.0308(3)	-155.0205	-	-155.0545	-154.1573	-	97.36(4)
Bicyclobutane	-	-155.9263(6)	-155.9216	-	-155.9575	-154.9372	-	96.94(5)







# RunFlow

- Add _user.py file with your details
    - user = your server user id




# Doc
## How to Run
###### #pyfig-recurse
Calling properties recursively is bad, exclude some properties from config dictionary to avoid this. 

###### #_n_submit-state-flow
initial -1 ( into pyfig.submit() )
changed to number of submissions ( into pyfig.__init__, is positive 1 or n_sweep, exits after submission )
changed to zero (on cluster submit)


- hypam opt
    - opt: adahess, radam
    - cudnn: T, F
    - lr: sample
    - hessian_power: sample
    - n_b: 1024

- hypam opt
    - n_sv
    - n_pv
    - n_fb
    - n_det


https://wandb.ai/hwat/hwat/groups/dist/workspace
https://wandb.ai/hwat/hwat/groups/sweep-a_z/workspace
https://wandb.ai/hwat/hwat/groups/sweep-n_b/workspace
https://wandb.ai/hwat/hwat/groups/hypam_opt/workspace

baselines :exclamation:
memory scaling and gpu usage scaling (1 gpu)
thread testing pytorch :exclamation:
optimise hyperparams
scale across 1 gpu n_e max_mem opt
scaling across gpus (naive)
scaling across gpus (accelerate)
(including nodes)
take those wandb reports and copy the application and send to eske
message Steen 8 node 4h block







# Docs
## Rules
### 


## How to run
### Definitions
- Local (your laptop or a cloud development environment)
- Server (the stepping stone to cluster)
- Cluster (where the magic happens)

### Commands
- jupyter nbconvert --execute <notebook>

### Local -> Server -> Cluster
- 

### Server -> Cluster
- 

# FIX 
- pre on diag gaussian
- Loop over dense
- Do arrays roll from right


# How To
- Fill in User Details in Pyfig


# Doc Notes
## Glossary


# Test Suite
- Anti-symmetry

# Theory notes
- Why is the mean over the electron axes equal to a quarter? 
    - The mean of the entire thing is equal to zero...
    - * this problem is interesting and lead to the tril idea

# Model Ideas
- Alter the dims of the embedded variables to match the first FermiBlock layer, giving more residual connections (ie mean with keepdims)
    - name: 'fb0_res'
    - r_s_res = r_s_var.mean(-1, keepdims=True) if _i.fb0_res else jnp.zeros((_i.n_e, _i.n_sv), dtype=r.dtype)
- Electrons are indistinguishable... Why no mix mas? Eg in the initial layers, extrac the mean out the fermi block and perform it every iteration removing the means from the fermi block 
- Tril to drop the lower triangle duh? 
    - Need to check the antisymmetry, for sure
- Max pools ARE ALSO PERMUTATION INVARIANT
- Keep the atom dimension so perform ops?
- To test 'only upper triangle' - tril the inputs
- Test limiting the log_psi esp early in training (regularise)


# Gist / Notion / Embed / Share
- https://blog.shorouk.dev/2020/06/how-to-embed-any-number-of-html-widgets-snippets-into-notion-app-for-free/

https://hwat.herokuapp.com/panel_demo


# Setup
## Requirements file
- pipreqsnb <jup-notebook>
- pipreqs <python-file>

## Procfile
### Abstract
- <indicate-what-kind-of-app-as-defined-by-heroku>: <a-cmd-to-run-the-app>
### Generalisation
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com
### Description
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com
### Example
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com

# Heroku
## tar install (NO MARCHA)
"""
wget https://cli-assets.heroku.com/branches/stable/heroku-OS-ARCH.tar.gz
tar -xvzf heroku-OS-ARCH -C /usr/local/lib/heroku
ln -s /usr/local/lib/heroku/bin/heroku /usr/local/bin/heroku
"""

## Launch app
- heroku login
- heroku create <app-name>
- git push heroku master
    - git push heroku main
- heroku create -a example-app [auto adds heroku remote]
- git remote -v (checks ]
- app_exists: 
    - heroku git:remote -a example-app
- git push heroku main

## ide-yeet 
- Other 


<script src="https://gist.github.com/xmax1/f9f66535467ec44759193a18594e72c4.js"></script>