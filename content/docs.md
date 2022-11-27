
# Pyfig

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

# Jax Sharp
class Classifier(nn.Module):
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.num_classes, name='head')(x)
    return x
num_classes = 3
model = Classifier(num_classes=num_classes)
x = jnp.ones((1, 224, 224, 3))
variables = model.init(jax.random.PRNGKey(1), x)
model.apply(variables, x) # this is important - I think it has something to do with the functionalitiness of Jax
# additionally it NEEDS the variables passed, which is usually handles by the train state ya


# Class FermiNet


# x (n_e, 3) sig (3, n_e * n_det) exp (n_e, n_e*n_det) ({e_0}^{n_e*n_det}, {e_1}}^{n_e*n_det}, ...)

# there should be a way to do this with init
  # do we have a baseline method? Always push to use pyfig? 
  # Can we use ** operator - if no, preinit everything, then options
  # Should indicate with * that everything is kw 
  # is the single electron feature graph invariant to sth
  """
    x (b, n_e, 3) e coord
    xu (b, n_u, 3) spin up e coord
    xd (b, n_d, 3) spin down e coord
    x_d :   : l1-norm e pos
    ee_disp :       : e-e displacement
    ee_d :       : e-e dist
    x_s:            : single stream variable
    x_p:    
    x_s_res:        : single stream residual 
    x_p_res: 

"""

# logabssumdet

Special case if there is only one electron in any channel
We can avoid the log(0) issue by not going into the log domain


# Objax

https://colab.research.google.com/github/google/objax/blob/master/docs/source/notebooks/Objax_Basics.ipynb
