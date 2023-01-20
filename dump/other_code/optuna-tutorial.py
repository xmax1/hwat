


import optuna
from typing import TypedDict


    

def gen_suggest(trial, v: Param):
    if v.domain:
        if v.dtype is int:
            suggest = lambda : trial.suggest_int(v.name, *v.range, log=v.log)
        elif v.dtype is float:
            suggest = lambda : trial.suggest_float(v.name, *v.range, log=v.log)
    else:
        suggest = trial.suggest_categorical(v.name, v.values)
    
    return suggest


def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Integer parameter
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Integer parameter (discretized)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Floating point parameter
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)