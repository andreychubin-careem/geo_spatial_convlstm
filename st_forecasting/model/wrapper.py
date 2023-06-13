import torch
import lightning as pl
from typing import Optional
from torch import nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional.regression.symmetric_mape import symmetric_mean_absolute_percentage_error
from torchmetrics.functional.regression.mape import mean_absolute_percentage_error
from torchmetrics.functional.regression.mse import mean_squared_error

from .losses import masked_loss
from .losses.rowwise_losses import mape, smape, mse


class ModelWrapper(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            scheduler: Optional[LambdaLR] = None,
            horizon: int = 1,
            loss_type: str = 'default',
            include_masked_metrics: bool = False,
            loss_fn_alias: str = 'mse',
            masked_weight: float = 0.7
    ):
        """
        Wrapper class for pl.Trainer.
        Distributed learning is not supported, but can be easily added.

        :param model: PyTorch Model to train
        :param optimizer: PyTorch optimizer
        :param scheduler: PyTorch scheduler
        :param horizon: forecasting horizon
        :param loss_type: what loss logic to use. ['default'|'masked'|'semi_masked']
        :param include_masked_metrics: whether to show metric values only for viable squares
        :param loss_fn_alias: alias for loss function ['smape'|'rmse'|'mse'|'mape'] (for 'rmse' MSE loss will be used)
        :param masked_weight: weight used for masked loss in 'semi_masked' training.
        [0.0, 1.0], where 1.0 is equal to fully masked loss
        """
        super(ModelWrapper, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.horizon = horizon
        self.use_masked_loss = loss_type == 'masked'
        self.use_semi_masked_loss = loss_type == 'semi_masked'
        self.include_masked_metrics = include_masked_metrics
        self.loss_fn_alias = loss_fn_alias
        self.masked_weight = masked_weight

        if not self.use_masked_loss:
            if self.loss_fn_alias == 'smape':
                self.basic_loss_fn = symmetric_mean_absolute_percentage_error
            elif self.loss_fn_alias == 'mape':
                self.basic_loss_fn = mean_absolute_percentage_error
            elif self.loss_fn_alias in ['rmse', 'mse']:
                if self.loss_fn_alias == 'rmse':
                    print('mse will be used instead of rmse as basic loss function')
                self.basic_loss_fn = mean_squared_error
            else:
                raise NotImplementedError(f'"{self.loss_fn_alias}" loss function is not implemented')

        if self.use_masked_loss or self.use_semi_masked_loss:
            if self.loss_fn_alias == 'smape':
                self.loss_fn = smape
            elif self.loss_fn_alias == 'mape':
                self.loss_fn = mape
            elif self.loss_fn_alias in ['rmse', 'mse']:
                if self.loss_fn_alias == 'rmse':
                    print('mse will be used instead of rmse as masked loss function')
                self.loss_fn = mse
            else:
                raise NotImplementedError(f'"{self.loss_fn_alias}" loss function is not implemented')

        self.metric_1 = mean_squared_error
        self.metric_2 = mean_absolute_percentage_error

    @staticmethod
    def _get_mask(x: Tensor) -> Tensor:
        return torch.where(torch.sum(x, dim=1) > 0.0, 1.0, 0.0).unsqueeze(dim=1).to(x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        output = self.model(x, horizon=self.horizon)
        return output

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch

        # TODO: move it to loader
        mask = self._get_mask(x)

        y_hat = self.forward(x)

        if self.use_masked_loss:
            loss = masked_loss(y_hat, y, mask, self.loss_fn)
        elif self.use_semi_masked_loss:
            m_loss = masked_loss(y_hat, y, mask, self.loss_fn)
            d_loss = self.basic_loss_fn(y_hat, y)
            loss = torch.add(self.masked_weight * m_loss, (1.0 - self.masked_weight) * d_loss)
        else:
            loss = self.basic_loss_fn(y_hat, y)

        rmse_value = self.metric_1(y_hat, y, squared=False)
        mape_value = self.metric_2(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mape', mape_value, on_step=False, on_epoch=True, prog_bar=True)

        if self.include_masked_metrics:
            m_rmse_value = torch.sqrt(masked_loss(y_hat, y, mask, mse))
            m_mape_value = masked_loss(y_hat, y, mask, mape)
            self.log('train_rmse(m)', m_rmse_value, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_mape(m)', m_mape_value, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, y = batch

        # TODO: move it to loader
        mask = self._get_mask(x)

        y_hat = self.forward(x)

        if self.use_masked_loss:
            loss = masked_loss(y_hat, y, mask, self.loss_fn)
        elif self.use_semi_masked_loss:
            m_loss = masked_loss(y_hat, y, mask, self.loss_fn)
            d_loss = self.basic_loss_fn(y_hat, y)
            loss = torch.add(self.masked_weight * m_loss, (1.0 - self.masked_weight) * d_loss)
        else:
            loss = self.basic_loss_fn(y_hat, y)

        rmse_value = self.metric_1(y_hat, y, squared=False)
        mape_value = self.metric_2(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mape', mape_value, on_step=False, on_epoch=True, prog_bar=True)

        if self.include_masked_metrics:
            m_rmse_value = torch.sqrt(masked_loss(y_hat, y, mask, mse))
            m_mape_value = masked_loss(y_hat, y, mask, mape)
            self.log('val_rmse(m)', m_rmse_value, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_mape(m)', m_mape_value, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        if self.scheduler is None:
            return {
                'optimizer': self.optimizer,
                'monitor': 'val_loss'
            }
        else:
            return {
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'monitor': 'val_loss'
            }
