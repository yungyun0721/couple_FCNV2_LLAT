from os import system
from time import time
from pathlib import Path
from builtins import ValueError

import lightning.pytorch as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from utils.types import Shape_T

from .datasets import ERA5TCDataset


class ERA5TCDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str = "",
        data_tar: str | Path = "",
        fast_dir: str | Path = "",
        n_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        multiprocessing_context: str | None = None,
        data_spatial_shape: Shape_T = (13, 81, 81),
        train_start: int = 2007,
        train_end: int = 2018,
        val_start: int = 2018,
        val_end: int = 2020,
        test_start: int = 2020,
        test_end: int = 2021,
        combined_nc_input: bool = False,
        stat_mean_file: str = 'stat_mean.nc',
        stat_std_file: str = 'stat_std.nc',
        stat_mean_upper_file: str = 'stat_2007_mean_upper.nc',
        stat_std_upper_file: str = 'stat_2007_std_upper.nc',
        stat_mean_sfc_file: str = 'stat_2007_mean_sfc.nc',
        stat_std_sfc_file: str = 'stat_2007_std_sfc.nc',
        upper_variables: list = ["u", "v", "t", "q", "z"],
        surface_variables: list = ["u10", "v10", "t2m", "msl"],
        batch_size: int = 1,
        inferencing: bool = False,
        ingest_space_info: bool = False,
    ) -> None:
        """
        Data module for ERA5TW dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            train_start (str, optional): Start time of the training set. Defaults to "2020-07-01 00:00:00".
            train_end (str, optional): End time of the training set. Defaults to "2020-07-21 00:00:00".
            val_start (str, optional): Start time of the validation set. Defaults to "2020-07-21 00:00:00".
            val_end (str, optional): End time of the validation set. Defaults to "2020-07-26 00:00:00".
            test_start (str, optional): Start time of the test set. Defaults to "2020-07-26 00:00:00".
            test_end (str, optional): End time of the test set. Defaults to "2020-08-01 00:00:00".
            upper_variables (list, optional): Upper level variables. Defaults to ["u", "v", "t", "q", "z", "w"].
            surface_variables (list, optional): Surface level variables. Defaults to ["u10", "v10", "t2m", "msl", "sp", "tcwv", "tp", "d2m"].
            batch_size (int, optional): Batch size. Defaults to 32.
            n_workers (int, optional): Number of workers for dataloader. Defaults to 4.

        Note:
            The time format is "%Y-%m-%d %H:%M:%S".
            End time is exclusive.
            Time step is 1 hour.
        """
        super().__init__()
        self.root_dir: str = root_dir
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end

        self.stat_mean_file = stat_mean_file
        self.stat_std_file = stat_std_file
        self.stat_mean_upper_file = stat_mean_upper_file
        self.stat_std_upper_file = stat_std_upper_file
        self.stat_mean_sfc_file = stat_mean_sfc_file
        self.stat_std_sfc_file = stat_std_sfc_file

        self.data_spatial_shape = data_spatial_shape
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

        self.combined_nc_input = combined_nc_input

        self.inferencing = inferencing
        self.ingest_space_info = ingest_space_info

        available_root = root_dir and Path(root_dir).is_dir()
        if not fast_dir and not available_root:
            raise ValueError(
                "Provide a vaild root_dir or provide fast_dir with or without data_tar."
            )

        if not fast_dir:
            # use root_dir for reading data
            return

        fast_dir = Path(fast_dir)
        if fast_dir.exists() and fast_dir.is_dir():
            # use fast_dir for reading data
            print(
                f"{fast_dir} already existed, using it as the root directory...",
                flush=True,
            )
            self.root_dir = str(fast_dir)
            return

        try:
            fast_dir.mkdir(parents=True, exist_ok=False)
        except:
            print("Cannot create fast_dir, fallback to root_dir...", flush=True)
            return

        if not fast_dir.is_dir():
            print(f"Cannot access {fast_dir}, fallback to root_dir...", flush=True)
            return

        if data_tar:
            data_tar = Path(data_tar)
            if data_tar.is_file():
                print(
                    f"Starting to extract data from {data_tar} to {fast_dir}...",
                    flush=True,
                )
                timestamp_start = time()
                system("date")
                if not system(f"tar xf {data_tar} -C {fast_dir}"):
                    system("date")
                    timestamp_stop = time()
                    print(
                        f"Extract finished in {timestamp_stop-timestamp_start} secs.",
                        flush=True,
                    )
                    self.root_dir = str(fast_dir)
                else:
                    print("Failed to extract data_tar, fallback...", flush=True)
                return
            else:
                print("Cannot access data_tar, fallback...", flush=True)

        if available_root:
            print(f"Starting to copy data from {root_dir} to {fast_dir}...", flush=True)
            timestamp_start = time()
            system("date")
            if not system(f"cd {root_dir}; tar cf - . | (cd {fast_dir} && tar xf -)"):
                system("date")
                timestamp_stop = time()
                print(
                    f"Copy finished {timestamp_stop-timestamp_start} secs.", flush=True
                )
                self.root_dir = str(fast_dir)
            else:
                print(f"Failed to copy data to {fast_dir}, fallback...", flush=True)
        else:
            raise ValueError(f"Cannot access {root_dir}, failed to start...")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        self.train_dataset = ERA5TCDataset(
            self.root_dir,
            self.train_start,
            self.train_end,
            self.upper_variables,
            self.surface_variables,
            self.data_spatial_shape,
            combined_nc_input=self.combined_nc_input,
            standardize=True,
            stat_mean_file=self.stat_mean_file,
            stat_std_file=self.stat_std_file,
            stat_mean_upper_file=self.stat_mean_upper_file,
            stat_std_upper_file=self.stat_std_upper_file,
            stat_mean_sfc_file=self.stat_mean_sfc_file,
            stat_std_sfc_file=self.stat_std_sfc_file,
            get_stat=False,
            inferencing=self.inferencing,
            ingest_space_info=self.ingest_space_info,
        )
        self.val_dataset = ERA5TCDataset(
            self.root_dir,
            self.val_start,
            self.val_end,
            self.upper_variables,
            self.surface_variables,
            self.data_spatial_shape,
            combined_nc_input=self.combined_nc_input,
            standardize=True,
            stat_mean_file=self.stat_mean_file,
            stat_std_file=self.stat_std_file,
            stat_mean_upper_file=self.stat_mean_upper_file,
            stat_std_upper_file=self.stat_std_upper_file,
            stat_mean_sfc_file=self.stat_mean_sfc_file,
            stat_std_sfc_file=self.stat_std_sfc_file,
            get_stat=True,
            inferencing=self.inferencing,
            ingest_space_info=self.ingest_space_info,
        )
        self.test_dataset = ERA5TCDataset(
            self.root_dir,
            self.test_start,
            self.test_end,
            self.upper_variables,
            self.surface_variables,
            self.data_spatial_shape,
            combined_nc_input=self.combined_nc_input,
            standardize=True,
            stat_mean_file=self.stat_mean_file,
            stat_std_file=self.stat_std_file,
            stat_mean_upper_file=self.stat_mean_upper_file,
            stat_std_upper_file=self.stat_std_upper_file,
            stat_mean_sfc_file=self.stat_mean_sfc_file,
            stat_std_sfc_file=self.stat_std_sfc_file,
            get_stat=True,
            inferencing=self.inferencing,
            ingest_space_info=self.ingest_space_info,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
            num_workers=self.n_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
            num_workers=self.n_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=self.n_workers
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError
