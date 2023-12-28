import torch
from typing import List


class residual_layer(torch.nn.Module):

    def __init__(self, d_in: int, d_out:int):
        """
        :param d_in: input dim of linear layer
        :param d_out: output dim of linear layer
        """
        super().__init__()
        self.fc = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x) + x


class deep_linear_mdl(torch.nn.Module):

    def build_compressor(self, d:int, n_layers:int, residuals:List[bool], reverse:bool=False):
        """
        :param d         : dimension of input
        :param n_layers  : the number of layers of encoder/decoder
        :param residuals : whether to replace a linear layer by a residual layer one 
        :param reverse   : forward is encoder and backward is decoder
        :returns: a linear model
        """
        seq = []
        for i in range(0, n_layers):
            in_dim = d/(2**i) if not reverse else d * (2**i)
            out_dim = d/(2**(i+1)) if not reverse else d * (2**(i+1))
            in_dim, out_dim = int(in_dim), int(out_dim)
            layer = torch.nn.Linear(in_dim, out_dim) if not residuals[i] else residual_layer(in_dim, out_dim)
            seq.append(layer)

        return torch.nn.Sequential(*seq)        

    
    def __init__(self, d:int, n_layers:int, residuals:List[bool]):

        super().__init__()
        self.encoder = self.build_compressor(d, n_layers, residuals)
        self.decoder = self.build_compressor(d/(2**n_layers), n_layers, residuals, True)

    def forward(self, x):
        return self.decoder(self.encoder(x))

