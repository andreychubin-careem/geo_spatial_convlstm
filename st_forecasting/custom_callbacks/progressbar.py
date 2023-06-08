from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


def progress_bar():
    return RichProgressBar(
        theme=RichProgressBarTheme(
            description="black",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="black",
            time="black",
            processing_speed="black",
            metrics="black",
        ),
        leave=True
    )
