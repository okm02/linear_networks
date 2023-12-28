import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
# custom scripts
from data import build_dataset 
from utils import load_model, fetch_config
# typing
from typing import Dict


def extract_norms(mdl: torch.nn.Module):
    """
    :param mdl: model with pretrained weights
    :returns: dictionary with spectral norm computed across all model weights
    """
    spec_norms = {'encoder':[], 'decoder':[]}
    for i in range(0, len(mdl.encoder)):
        spec_norms['encoder'].append(torch.linalg.svdvals(mdl.encoder[i].weight)[0].item())
        spec_norms['decoder'].append(torch.linalg.svdvals(mdl.decoder[i].weight)[0].item())
    return spec_norms


def plot_norms(norm_dct:Dict, plt_name:str):
    """
    :param norm_dct: dictionary mapping each model components to the spectral norms of each layer
    :param mdl_depth: depth of linear/encoder
    """    
    d_range = list(range(0, len(norm_dct['encoder'])))
    
    fig = plt.Figure(figsize=(12, 8))
    plt.plot(d_range, norm_dct['encoder'], label='encoder')
    plt.plot(d_range, norm_dct['decoder'], label='decoder')
    plt.grid()
    plt.legend()
    plt.xlabel('Depth')
    plt.ylabel('Spectral norm')
    plt.title('Norms of layers across model')
    plt.savefig(f'plots/{plt_name}.png')
    plt.show()


def compute_precision(mdl:torch.nn.Module, R: torch.Tensor) -> None:
    """
    :param R: the rotation matrix used to perturb the input generating the labels
    :param mdl: the deep linear model
    :param depth: depth of encoder/decoder
    """  
    # extract weight matrices of all layers
    depth = len(mdl.encoder)
    prod_wts = [None] * (2 * depth)
    for i in range(0, depth):
        prod_wts[i] = mdl.encoder[i].weight.T
        prod_wts[i+depth] = mdl.decoder[i].weight.T

    # compute product of all linear layers (the reconstruction of the rotation matrix)
    R_prime = torch.linalg.multi_dot(prod_wts)

    # compute equivalence of both matrices
    precision = torch.nn.functional.l1_loss(R_prime, R).item()

    print(f'Factorization of rotation matrix with precision = {precision}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mdl_path', type=str, help='path to model weights')
    args = parser.parse_args()

    # initialize model
    mdl = load_model(cpath='hp.json', wts_path=args.mdl_path)
    
    # build rotation matrix
    conf = fetch_config('hp.json')
    _, R = build_dataset(N=conf['dataset']['size'], d=conf['dataset']['dim'], sigma=conf['dataset']['variance'], seed=conf['seed'])

    # run analysis
    plot_norms(norm_dct=extract_norms(mdl=mdl), plt_name='solution_norms')

    compute_precision(mdl=mdl, R=R)
