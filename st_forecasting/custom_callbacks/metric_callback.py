import copy
import lightning as pl


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)
