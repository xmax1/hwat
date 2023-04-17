import os

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from pydantic import create_model
from typing import Dict
from things.core import lo_ve

import optree
import numpy as np
from typing import Callable
import torch
from pprint import pprint

def run_dash(pyfig_d: dict) -> dict:

    app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

    cptForm: str = 'form-control mb-3'
    finBtn = "btn btn-primary gap-2 col-1 mx-auto"
    cellWidth = '3'
    cptForm = 'form-control mb-3'
    supported_types = ['string', 'integer', 'boolean', 'array', 'float']
    # "text", 'number', 'password', 'email', 'range', 'search', 'tel', 'url', 'hidden'; optional
    dash_supported_types: list[str] = ['text', 'number', 'checkbox', 'number']

    def get_dash_input_type(value_type: str) -> str:
        return dict(
            string='text',
            integer='number',
            boolean='checkbox',
            array='text',
            float='number',
        ).get(value_type)

    def get_element(key: str, info: dict, cpt_id: int) -> Component:

        value = info.get('default')
        value_type = info.get('type')
        
        if value_type is None:
            print(f'No type found for {key}, setting as text')
            value_type = 'text'
        
        dash_type = get_dash_input_type(value_type)

        if dash_type is None: 
            print(f'Potential unsupported dash type {dash_type} for {key}')
            dash_type = 'None'
        
        # Create Elements
        title_element = dbc.Label(key + f' ({value_type}) [dash_type: {dash_type}]')

        if dash_type == 'checkbox':
            value_element = dbc.RadioItems(
                options=['True', 'False'], value= value, 
                className= cptForm, id= key + str(cpt_id),
            )
        elif dash_type in ['text', 'number', 'None']:
            value_element = dbc.Input(
                value= value, type= dash_type, 
                className= cptForm, id= key + str(cpt_id),
            )
        else:
            print('value_type: ', value_type)
            raise ValueError(f'Unsupported dash_type {dash_type} for {key}')

        cpt = dbc.Col(width= cellWidth, children=[title_element, value_element])
        return cpt

    def _make_model(v, name):
        if type(v) is dict:
            return create_model(name, **{k: _make_model(v, k) for k, v in v.items()}), ...
        return type(v), v

    def make_model(v: Dict, name: str):
        return _make_model(v, name)[0]


    elements = dict()
    models = dict()

    def gather_models(name: str, schema: dict):
            
        base_model_d = dict()
        for key, val in schema.items():
            if isinstance(val, dict):
                gather_models(key, val)
            else:
                base_model_d[key] = val
        model = make_model(base_model_d, name)
        model = model.parse_obj(schema)
        models[key] = model
                
    print('launching dash app')
    # pyfig_d: BaseModel = dict_model('pyfig_d', pyfig_d)
    pyfig_d = optree.tree_map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v, pyfig_d)
    pyfig_d = optree.tree_map(lambda v: 'N/A' if isinstance(v, Callable) else v, pyfig_d)
    pyfig_d = optree.tree_map(lambda v: 'N/A' if isinstance(v, torch.dtype) else v, pyfig_d)
    
    gather_models('pyfig', pyfig_d)

    for name, configuration in models.items():

        elements[name] = html.H1(name, style={'color': '#ccccccc', 'textAlign': 'center'})

        schema = configuration.schema()

        for key, val in schema['properties'].items():
            cpt_id = len(elements)
            elements[key] = get_element(key, val, cpt_id)

    page = html.Div(
            dbc.Row(
        [
        dbc.Form(children=[dbc.Row(list(elements.values()))], id= "main-form"),
        html.A(
            dbc.Button('fin', className=finBtn, color='primary', id='button-submit', n_clicks= 0), 
            href='/'
        ),
        html.Div(id='body-div')
        ]), 
        className= 'p-5'
    )

    track = [
        Input(component_id= 'button-submit', component_property="n_clicks"),
    ]

    input_values = list()

    for key, el in elements.items():
        if hasattr(el, 'children'):
            for child in el.children:
                if isinstance(child, dbc.Input):
                    input_values += [child]
                    track += [Input(component_id= child.id, component_property="value")]

    print('Tracking {n} elements'.format(n= len(track)))

    app.layout = html.Div(page)

    def exit_app():
        print('Exiting')
        os._exit(0)

    @app.callback(Output('body-div', 'children'), track)
    def submit_message(fin_button_clicks, *out_data):
        print('fin_button_clicks: ', fin_button_clicks)
        if fin_button_clicks:
            c_update = dict()
            print(elements)
            for k, v in zip(elements.keys(), out_data):
                print(k, v)
                c_update[k] = v
            print(c_update)
            lo_ve(data=c_update, path='dash.yaml')
            exit_app()

    return app

# 
# d = {'a': 1, 'b': 2, 'c': 3} 
# dashboard = run_dash(d)
# 

if __name__ == '__main__':   
    dashboard.run_server()

    # test = {
    #     'app': {
    #         'a': [[0., 0., 0.]],
    #         'a_z': [4],
    #         'acc_target': 0.5,
    #         'charge': 0,
    #         'compute_energy': False,
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'init_data_scale': 1.0,
    #         'loss': '',
    #         'mo_coef': None,
    #         'mol': None,
    #         'n_corr': 20,
    #         'n_d': 2,
    #         'n_e': 4,
    #         'n_equil_step': 500,
    #         'n_u': 2,
    #         'spin': 0,
    #         'system_id': [[4, [0.0, 0.0, 0.0]]],
    #         'system_id_path': 'dump/[[4, [0.0, 0.0, 0.0]]].txt',
    #         'system_name': ''
    #     },
    #     'c_update_tag': 'c_update',
    #     'cudnn_benchmark': True,
    #     'data': {
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'loader_n_b': 1,
    #         'n_b': 128
    #     },
    #     'data_tag': 'data',
    #     'debug': True,
    #     'device': '',
    #     'dist': {
    #         'dist_id': 'Fail-',
    #         'dist_name': 'Naive',
    #         'gpu_id': 'Fail',
    #         'head': False,
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'launch_cmd': None,
    #         'n_launch': 1,
    #         'n_worker': 1,
    #         'nsync': 0,
    #         'pid': -1,
    #         'rank': -1,
    #         'rank_env_name': 'RANK',
    #         'ready': True,
    #         'sync_every': 5
    #     },
    #     'dtype': 'torch.float32',
    #     'entity': 'xmax1',
    #     'env': 'zen',
    #     'eval_tag': 'eval',
    #     'exp_id': '6550588',
    #     'exp_name': 'junk-4',
    #     'gather_tag': 'gather',
    #     'group_exp': False,
    #     'group_i': 0,
    #     'ignore_p': ['parameters', 'scf', 'tag', 'mode_c'],
    #     'is_logging_process': True,
    #     'lo_ve_path': '',
    #     'lo_ve_path_tag': 'lo_ve_path',
    #     'logger': {
    #         'entity': 'xmax1',
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'job_type': '',
    #         'log_mode': 'online',
    #         'log_run_path': '',
    #         'log_run_type': 'groups/junk-4/workspace',
    #         'log_run_url': 'https://wandb.ai/xmax1/hwat-tutorial/groups/junk-4/workspace',
    #         'program': 'projects/hwat-tutorial/run.py',
    #         'run': None
    #     },
    #     'max_mem_alloc_tag': 'max_mem_alloc',
    #     'mean_tag': 'mean',
    #     'mode': '',
    #     'model': {
    #         'compile_func': False,
    #         'compile_ts': False,
    #         'functional': True,
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'ke_method': 'grad_grad',
    #         'n_det': 4,
    #         'n_fb': 3,
    #         'n_fbv': 160,
    #         'n_final_out': 1,
    #         'n_pv': 32,
    #         'n_sv': 32,
    #         'optimise_aot': False,
    #         'optimise_ts': False,
    #         'terms_p_emb': ['rr', 'rr_len'],
    #         'terms_s_emb': ['ra', 'ra_len'],
    #         'with_sign': False},
    # 'multimode': '',
    # 'n_default_step': 10,
    # 'n_eval_step': 0,
    # 'n_log_metric': 50,
    # 'n_log_state': 1,
    # 'n_max_mem_step': 0,
    # 'n_opt_hypam_step': 0,
    # 'n_pre_step': 0,
    # 'n_step': 10,
    # 'n_train_step': 0,
    # 'opt': {'available_opt': ['AdaHessian',
    #                         'RAdam',
    #                         'Apollo',
    #                         'AdaBelief',
    #                         'LBFGS',
    #                         'Adam',
    #                         'AdamW'],
    #         'beta': 0.9,
    #         'betas': (0.9, 0.999),
    #         'eps': 0.0001,
    #         'hessian_power': 1.0,
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'init_lr': 0.001,
    #         'lr': 0.001,
    #         'opt_name': 'AdamW',
    #         'warm_up': 100,
    #         'weight_decay': 0.0},
    # 'opt_hypam_tag': 'opt_hypam',
    # 'opt_obj_all_tag': 'opt_obj_all',
    # 'opt_obj_key': 'e',
    # 'opt_obj_tag': 'opt_obj',
    # 'paths': {
    #     'cluster_dir': 'dump/exp/junk-4/6550588/cluster',
    #     'code_dir': 'dump/exp/junk-4/6550588/code',
    #     'dump_dir': 'dump',
    #     'dump_exp_dir': 'dump/exp',
    #     'exchange_dir': 'dump/exp/junk-4/6550588/exchange',
    #     'exp_data_dir': 'dump/exp/junk-4/6550588/exp_data',
    #     'exp_dir': 'dump/exp/junk-4/6550588',
    #     'home': '.',
    #     'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #     'project_dir': 'projects/hwat-tutorial',
    #     'state_dir': 'dump/exp/junk-4/6550588/state',
    #     'tmp_dir': 'dump/tmp',
    #     'pre_tag': 'pre',
    #     'project': 'hwat-tutorial',
    #     'resource': {
    #         'architecture': 'cuda',
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'job_id': 'No SLURM_JOBID available.',
    #         'n_device': 0,
    #         'n_device_env': 'CUDA_VISIBLE_DEVICES',
    #         'n_gpu': 1,
    #         'n_node': 1,
    #         'n_thread_per_process': 8,
    #         'nifl_gpu_per_node': 10,
    #         'pci_id': 'Fail',
    #         'slurm_c': {
    #             'cpus_per_gpu': 8,
    #             'error': 'dump/exp/junk-4/6550588/cluster/e-%j.err',
    #             'export': 'ALL',
    #             'gres': 'gpu:RTX3090:1',
    #             'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #             'job_name': 'junk-4',
    #             'nodes': '1',
    #             'ntasks': 1,
    #             'output': 'dump/exp/junk-4/6550588/cluster/o-%j.out',
    #             'partition': 'sm3090',
    #             'time': '0-00:10:00'
    #         }
    #     },
    #     'run_debug_c': False,
    #     'run_id': '',
    #     'run_name': 'run.py',
    #     'run_sweep': False,
    #     'scheduler': {
    #         'gamma': 1.0,
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'sch_epochs': 1,
    #         'sch_gamma': 0.9999,
    #         'sch_max_lr': 0.01,
    #         'sch_name': 'ExponentialLR',
    #         'sch_verbose': False,
    #         'scheduler_name': '',
    #         'step_size': 1
    #     },
    #     'seed': 808017424,
    #     'submit': False,
    #     'sweep': {
    #         'ignore_base': ['ignore', 'p', 'd', 'd_flat', 'ignore_base'],
    #         'n_trials': 20,
    #         'storage': 'sqlite:///dump/exp/junk-4/6550588/hypam_opt.db',
    #         'sweep_name': 'study'
    #     },
    #     'train_tag': 'train',
    #     'user': 'xmax1',
    #     'v_cpu_d_tag': 'v_cpu_d',
    #     'zweep': ''
    # }}

