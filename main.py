import os
import hydra
import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from src.data.data_loader import RealStudentLoader, CircleStudentLoader
from src.heuristic.vns import VNSGroupOptimizer, ExactOptimizer
from src.analysis.stats import print_grouping_stats
from src.utils.save import save_run_data

def setup_optimizer(data, config):
    """Initialize VNS optimizer with config parameters"""

    optimizer_name = config['optim']['name']
    if optimizer_name == 'vns':
        optimize_params = {'max_optim_steps': config['optim']['params']['max_optim_steps'],
                           'max_no_improve': config['optim']['params']['max_no_improve'],
                           'max_local_search_tries': config['optim']['params']['max_local_search_tries'],
                           'return_groups_only': False}
        return VNSGroupOptimizer(
            q=data['q'],
            s=data['s'],
            min_group_size=config['optim']['params']['min_group_size'],
            max_group_size=config['optim']['params']['max_group_size'],
            lower_balance=config['optim']['params']['balance'],
            seed=config['optim']['params']['seed'],
            maximize=config['optim']['params']['maximize'],
            pvns=config['optim']['params']['pvns'],
            space_n_sample=config['optim']['params']['space_n_sample'],
            balance_reg_weight=config['optim']['params']['balance_reg_weight'],
            size_reg_weight=config['optim']['params']['size_reg_weight']
            ), optimize_params

    optimize_params = {'max_time': None,
                       'return_groups_only': False}
    return ExactOptimizer(
        q=data['q'],
        s=data['s'],
        min_group_size=config['optim']['params']['min_group_size'],
        max_group_size=config['optim']['params']['max_group_size'],
        lower_balance=config['optim']['params']['balance'],
        seed=config['optim']['params']['seed'],
        maximize=config['optim']['params']['maximize']), optimize_params

def get_data_loader(config):
    """Get data loader based on config"""

    if config['data']['name'] == 'real':
        return RealStudentLoader(config['data'])
    elif config['data']['name'] == 'circle':
        return CircleStudentLoader(config['data'])
    else:
        raise NotImplementedError('Data loader not implemented.')

@hydra.main(version_base=None, config_path="configs", config_name="config_circle_sweep")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    hydra_cfg = HydraConfig.get()
    save_path = hydra_cfg.runtime.output_dir

    if type(cfg.optim.params.maximize) == str:
        maximize_str = cfg.optim.params.maximize
        cfg.optim.params.maximize = True if maximize_str == 'True' else False if maximize_str == 'False' else None

    if type(cfg.optim.params.space_n_sample) == str:
        space_n_sample = cfg.optim.params.space_n_sample
        cfg.optim.params.space_n_sample = None if space_n_sample == 'None' else int(space_n_sample)
        
    # load and process all data
    data_loader = get_data_loader(cfg)
    data = data_loader.load_all()
    
    # optimize
    o, optimize_params = setup_optimizer(data, cfg)
    best_groups, (best_solution, best_cost), trajectory_df = o.optimize(**optimize_params)
    
    # save results
    print_grouping_stats(best_groups, best_cost, data, cfg, save_path)
    save_run_data(data, best_groups, trajectory_df, cfg, save_path)
    return best_cost

if __name__ == "__main__":
    main()