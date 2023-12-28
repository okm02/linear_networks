import torch, json
from typing import Dict
from model import deep_linear_mdl


def fetch_config(cpath:str) -> Dict:
    """
    :param cpath: path having configuration
    :returns: configuration loaded into dictionary
    """
    with open(cpath, 'r') as f:
        return json.load(f)


def load_pretrained_wts(path: str) -> Dict[str, torch.Tensor]:
    """
    :param path: path having checkpoint
    :returns: loaded pretrained weights
    """
    wts = torch.load(path)['state_dict']
    wts = {k.replace('mdl.', ''):v for k, v in wts.items()}
    return wts


def init_model(conf: str) -> torch.nn.Module:
    """
    :param conf: the configuration of the experiment
    :returns: the deep linear model initialized
    """
    mdl = deep_linear_mdl(d=conf['dataset']['dim'], n_layers=conf['architecture']['depth'],
                          residuals=[conf['architecture']['residuals']]*conf['architecture']['depth']*2)
    return mdl


def load_model(cpath: str, wts_path:str) -> torch.nn.Module:
    """
    :param cpath: path to experiment configuration
    :param wts_path : path having model checkpoints
    :returns: model with pretrained weights loaded
    """
    # load configuration and initialize model
    config = fetch_config(cpath)
    mdl = init_model(conf=config)
    # load pretrained weights from checkpoint
    wts = load_pretrained_wts(wts_path)
    # load weights into the model
    mdl.load_state_dict(wts)
    
    return mdl
    