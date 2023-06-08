from lightning.pytorch.callbacks import ModelCheckpoint


def checkpoint_callback():
    return ModelCheckpoint(
        monitor='val_loss',
        save_top_k=10,
        every_n_epochs=1,
        auto_insert_metric_name=True
    )
