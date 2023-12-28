import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, Dict


def gen_data(N:int, d:int, sigma: Union[float, torch.Tensor], seed:int=42) -> torch.Tensor:
    """
    :param N : number of data points
    :param d : data dimension
    :param sigma : variance per feature
    :returns: gaussian samples (isotropic/non-isotropic)
    """

    np.random.seed(seed)
    # generate parameter tensors
    mu, cov = np.zeros(d), np.ones(d) * sigma
    cov = np.diag(cov)
    # build a gaussian sampler
    dist = np.random.multivariate_normal(mu, cov, size=N)
    
    return torch.from_numpy(dist).type(torch.float32)


def build_labels(X:torch.Tensor, seed:int=42) -> torch.Tensor:
    """
    :param X: data inputs
    :returns: a rotation of the matrix
    """

    def gen_psd_matrix(d: int) -> torch.Tensor:
        """
        :param d: data dimension
        :returns: symmetric positive definite matrix to rotate data
        """
        rseed_generator = torch.Generator()
        rseed_generator.manual_seed(seed)
        rm = torch.rand((d, d), generator=rseed_generator)
        R = torch.mm(rm.T, rm)
        # scale directions to unit norm
        min_, max_ = torch.min(R, 1)[0], torch.max(R, 1)[0]
        R_scale = (R - min_)/(max_ - min_)
        return R_scale.T

    rot = gen_psd_matrix(d=X.shape[1])
    return torch.mm(X, rot), rot


def build_dataset(**kwargs):

    X = gen_data(N=kwargs['N'], d=kwargs['d'], sigma=kwargs['sigma'], seed=kwargs['seed'])
    y, R = build_labels(X, seed=kwargs['seed'])

    return (X, y), R


def split_data(X: torch.Tensor, y: torch.Tensor, test_size:float=0.1, val_size:float=0.05, seed:int=42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    :param X: input features
    :param y: target prediction matrix
    :returns: train,val, test splits
    """

    idx = list(range(0, X.shape[0]))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    train_idx, val_idx  = train_test_split(train_idx, test_size=val_size, random_state=seed, shuffle=True)
    
    dataset = TensorDataset(X, y)
    train_set, val_set, test_set = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    return (train_set, val_set), test_set


def get_loader_fn(d_set: torch.utils.data.Dataset, B:int, inf_:bool=False):
    """
    :param d_set: a torch dataset
    :param B : number of samples in batch
    :param inf_: a boolean denoting of this subsample is a dataset where inference needs to be performed
    :returns: iterator to pass over the data
    """
    return DataLoader(d_set, batch_size=B, shuffle=not inf_, num_workers=4)


def build_torch_sets(conf: Dict):
    """
    :param conf: a configuration dict having all hyper-parameters of experiment
    This function serves as a wrapper around different functions to construct a dataset
    and wrap it with torch's dataset wrapper
    """
    # generate the data
    (X, y), R = build_dataset(N=conf['dataset']['size'], d=conf['dataset']['dim'], sigma=conf['dataset']['variance'], seed=conf['seed'])
    # generate the splits
    return split_data(X, y, conf['dataset']['test_size'], conf['dataset']['val_size'], conf['seed'])
