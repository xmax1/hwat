

# TODO
- 

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
- Pretrain on diag gaussian
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