import torch, json
import lightning as pl
from typing import List, Dict
# custom scripts
from model import deep_linear_mdl
from data import build_torch_sets, get_loader_fn
from utils import fetch_config, init_model


class PL_Deep_Linear(pl.LightningModule):

    def __init__(self, mdl: torch.nn.Module, optim_conf: Dict):
        """
        :param mdl : the model to be trained 
        """
        super().__init__()
        self.mdl = mdl
        self.optim_conf = optim_conf

    def step_(self, batch):
        
        x, y = batch
        y_reconst = self.mdl(x)
        loss = torch.nn.functional.mse_loss(y_reconst, y)
        return loss

    def training_step(self, batch, batch_idx):

        loss = self.step_(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
    
        loss = self.step_(batch)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim_conf['LR'])
        return optimizer


if __name__ == '__main__':

    # load experiment configuration
    cdict = fetch_config('hp.json')

    # build dataset
    (train_set, val_set), _ = build_torch_sets(cdict)
    train_loader = get_loader_fn(d_set=train_set, B=cdict['training']['batch_size'])
    val_loader   = get_loader_fn(d_set=val_set, B=cdict['training']['batch_size'], inf_=True)

    # build and train model
    mdl = PL_Deep_Linear(mdl=init_model(conf=cdict), optim_conf={'LR':cdict['training']['LR']})
    
    trainer = pl.Trainer(accelerator='gpu', devices=-1, num_nodes=1, max_epochs=cdict['training']['epochs'], enable_checkpointing=True, enable_progress_bar=True)
    trainer.fit(model=mdl, train_dataloaders=train_loader, val_dataloaders=val_loader)    
