import lightning as pl
from typing import Union, Iterable, Callable
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional.regression.mse import mean_squared_error

from .convttlstm.convttlstm import ConvTTLSTMNet


class ModelWrapper(pl.LightningModule):
    def __init__(
            self,
            model: ConvTTLSTMNet,
            optimizer: Optimizer,
            scheduler: LambdaLR = None,
            horizon: int = 1,
            loss_fn: Callable = mean_squared_error,
            metrics: Iterable = (),
    ):
        """
        Wrapper class for pl.Trainer.
        Distributed learning is not supported, but can be easily added.

        :param model: PyTorch Model to train
        :param optimizer: PyTorch optimizer
        :param scheduler: PyTorch scheduler
        :param horizon: forecasting horizon
        :param loss_fn:
        :param metrics:
        """
        super(ModelWrapper, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.horizon = horizon
        self.loss_fn = loss_fn
        self.metrics = metrics

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        output = self.model(x, mask=mask, horizon=self.horizon)
        return output

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, mask, y = batch
        y_hat = self.forward(x, mask)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, mask, y = batch
        y_hat = self.forward(x, mask)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Union[Optimizer, dict]:
        if self.scheduler is None:
            return self.optimizer
        else:
            return {
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'monitor': 'val_loss'
            }
