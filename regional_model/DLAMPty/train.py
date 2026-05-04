# from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import time
import torch
import lightning.__version__ as lightning_version
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from models.lightning_modules import PanguLightningModule
from utils.data_modules import ERA5TCDataModule
from typing_extensions import override

import torch

print(f"pytorch {torch.__version__}")
print(f"lightning {lightning_version}")

# torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision("high")


class ElapsedTimeCallback(Callback):
    @override
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Record the start time at the beginning of training
        self.start_time = time.time()

    @override
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Calculate elapsed time in seconds since training start
        elapsed = time.time() - self.start_time
        # Log the elapsed time (in seconds) as a metric;
        # this will make it available for checkpoint filename formatting
        pl_module.log("elapsed_time_totals", elapsed, prog_bar=False, logger=False)
        # (Optional) If you want to log minutes and seconds separately:
        days, seconds = divmod(int(elapsed), 86400)
        hours = seconds / 3600
        pl_module.log("elapsed_time_days", days, logger=False)
        pl_module.log("elapsed_time_hours", hours, logger=False)


def main():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=30,
        save_last='link',
        filename="{elapsed_time_days:02.0f}d-{elapsed_time_hours:04.1f}h-vl{val_loss:9.7f}-e{epoch}-s{step}",
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)

    cli = LightningCLI(
        model_class=PanguLightningModule,
        datamodule_class=ERA5TCDataModule,
        trainer_defaults={
            "profiler": "pytorch",
            "callbacks": [ElapsedTimeCallback(), checkpoint_callback, lr_monitor],
            # "callbacks": [checkpoint_callback, lr_monitor, device_stats_monitor]
        },
    )


if __name__ == "__main__":
    main()
