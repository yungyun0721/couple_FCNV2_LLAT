from os import path
from glob import glob
from tempfile import NamedTemporaryFile as ntf

import netCDF4
import numpy as np
import torch
import torch.utils.data

from .types import Stat_T, WeatherData
from utils.types import Shape_T
from utils.data_processor import combine_ncs_to_xarray

class ERA5TCDataset(torch.utils.data.Dataset):
    """
    Dataset for ERA5 reanalysis typhoon data.

    Returns:
        tuple[WeatherData, WeatherData]: Tuple of input and target weather data.
                                         One WeatherData is composed of upper-air data and surface data.
                                         Upper-air data is of shape (pressure level, latitude, longitude, variable).
                                         Surface data is of shape (latitude, longitude, variable).
    """

    def __init__(self,
                 root_dir: str,
                 start_year: int,
                 end_year: int,
                 upper_variables: list[str],
                 surface_variables: list[str],
                 data_spatial_shape: Shape_T,
                 combined_nc_input: bool = False,
                 stat_mean_file: str | None = None,
                 stat_std_file: str | None = None,
                 stat_mean_upper_file: str = 'stat_2007_mean_upper.nc',
                 stat_std_upper_file: str = 'stat_2007_std_upper.nc',
                 stat_mean_sfc_file: str = 'stat_2007_mean_sfc.nc',
                 stat_std_sfc_file: str = 'stat_2007_std_sfc.nc',
                 standardize: bool = False,
                 get_stat: bool = False,
                 inferencing: bool = False,
                 ingest_space_info: bool = False,
                 plugin_additional_vars: bool = False,
                 forecast_input: str = '',
                 ) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            start_year (int): Starting year of the ERA5 typhoon dataset.
            end_year (int): Ending year of the ERA5 typhoon dataset, excluded.
            upper_variables (list[str]): List of variables to be included in the upper-air data.
            surface_variables (list[str]): List of variables to be included in the surface data.
            standardize (bool, optional): Whether to standardize the data. Defaults to False.
            get_stat (bool, optional): Whether to return the mean and standard deviation of the dataset along with the data. This option is only effective when `standardize` is set to True. Defaults to False.
            data_spatial_shape (Shape_T, optional): Spatial shape of the data, expected to be (pressure levels, latitude, longitude). Defaults to (13, 161, 161).
            inferencing (bool, optional): Flag to indicate if the dataset is being used for inference purposes. Defaults to False.
        """
        super().__init__()
        self.inferencing = inferencing

        self.root_dir = root_dir
        self.data_spatial_shape = data_spatial_shape
        self.start_year = start_year
        self.end_year = end_year
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.ingest_space_info = ingest_space_info
        self.plugin_additional_vars = plugin_additional_vars

        self.combined_nc_input = combined_nc_input
        # self.stat_mean_upper_file = path.join('/nwpr/wfc/com136/data/ERA5_for_TC/1_WP',stat_mean_upper_file)
        # self.stat_std_upper_file = path.join('/nwpr/wfc/com136/data/ERA5_for_TC/1_WP',stat_std_upper_file)
        # self.stat_mean_std_file = path.join('/nwpr/wfc/com136/data/ERA5_for_TC/1_WP',stat_mean_std_file)
        # self.stat_std_std_file = path.join('/nwpr/wfc/com136/data/ERA5_for_TC/1_WP',stat_std_std_file)

        if forecast_input:
            cases = [f'forecast_{path.basename(forecast_input)}']
        else:
            cases = sorted(glob(f'{root_dir}/{start_year}/*'))
            for y in range(start_year+1,end_year):
                cases += sorted(glob(f'{root_dir}/{y}/*'))

        self.stat_mean_file = stat_mean_file
        self.stat_std_file = stat_std_file
        self.stat_mean_upper_file = stat_mean_upper_file
        self.stat_std_upper_file = stat_std_upper_file
        self.stat_mean_std_file = stat_mean_sfc_file
        self.stat_std_std_file = stat_std_sfc_file

        if forecast_input:
            if not combined_nc_input:
                print("Input file should be combined, and combined_nc_input should be True when doing forecast.")
            self.file0_list_combined = [forecast_input,]
            self.file1_list_combined = [forecast_input,]
        elif self.combined_nc_input:
            name_pattern = '*combined.nc'
            case_combined_files = [sorted(glob(f'{c}/{name_pattern}')) for c in cases]
            self.file0_list_combined = [file for a_case in case_combined_files for file in a_case[:-1]]
            self.file1_list_combined = [file for a_case in case_combined_files for file in a_case[1:]]
        else:
            name_pattern = '*upper.nc'
            case_upper_files = [sorted(glob(f'{c}/{name_pattern}')) for c in cases]
            self.file0_list_upper = [file for a_case in case_upper_files for file in a_case[:-1]]
            self.file1_list_upper = [file for a_case in case_upper_files for file in a_case[1:]]

        self.standardize = standardize
        if self.standardize:
            self.upper_mean, self.upper_std, self.surface_mean, self.surface_std = self._load_stat()
        self.get_stat = self.standardize and get_stat

        print(f'Combined input: {self.combined_nc_input}, Standardize: {self.standardize}, Get stat: {self.get_stat}')

    def _trim_var(self, input_var:netCDF4.Variable):
        """
        Trim the variable to the desired spatial shape.

        Args:
            input_var (netCDF4.Variable): Input variable to be trimmed.

        Returns:
            netCDF4.Variable: Trimmed variable.
        """
        trimed_var = input_var[:]
        x = input_var.shape[-1]
        y = input_var.shape[-2]
        if x != self.data_spatial_shape[2] or y != self.data_spatial_shape[1]:
            xd = int((x - self.data_spatial_shape[-1]) / 2)
            yd = int((y - self.data_spatial_shape[-2]) / 2)
            trimed_var = input_var[:][..., yd : y - yd, xd : x - xd]
        return trimed_var
    
    def _stat_from_nc(self, nc_file_path: str, want_sfc: bool) -> torch.Tensor:
        """
        Args:
            nc_file_path (str): Path to the netCDF file.
            want_sfc (bool): Whether to return the mean or standard deviation of the surface data.

        Returns:
            torch.Tensor: Mean or standard deviation of the upper-air data or surface data.
        """
        nc_path = nc_file_path
        if not path.exists(nc_file_path):
            nc_path = path.join(self.root_dir,nc_file_path)
        with netCDF4.Dataset(nc_path) as input_nc:
            var_list = self.upper_variables if not want_sfc else self.surface_variables
            stat_data = torch.stack([torch.from_numpy(input_nc[var][:]) for var in var_list], dim=-1)
            if want_sfc and len(stat_data.shape) == 4 or len(stat_data.shape) == 5:
                stat_data = stat_data.squeeze(-2)
            # when getting sfc stat and ingest_space_info=True, add `lon` and `lat` to the std or mean
            if want_sfc and self.ingest_space_info:
                # the stat data of lon and lat if we are loading std data
                c_lon, c_lat = (12, 12)
                # determine whether we are loading mean data by nc_file_path
                if nc_file_path.find('mean') != -1:
                    # we are loading mean data
                    c_lon, c_lat = (130, 15)
                shape = stat_data.shape[0:-1]
                lon = torch.zeros(shape)+c_lon
                lat = torch.zeros(shape)+c_lat
                stat_data = torch.cat((stat_data, lon.unsqueeze(-1), lat.unsqueeze(-1)), dim=-1)
            stat_data = torch.swapaxes(stat_data.float().cpu(),0,1)
        return stat_data

    def _load_stat(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of mean and std of upper-air data and surface data.
        """
        if self.combined_nc_input and self.stat_mean_file and self.stat_std_file:
            upper_mean = self._stat_from_nc(self.stat_mean_file, False)
            upper_std = self._stat_from_nc(self.stat_std_file, False)
            surface_mean = self._stat_from_nc(self.stat_mean_file, True)
            surface_std = self._stat_from_nc(self.stat_std_file, True)
            # print(upper_mean.shape, upper_std.shape, surface_mean.shape, surface_std.shape)
        else:
            upper_mean = self._stat_from_nc(self.stat_mean_upper_file, False)
            upper_std = self._stat_from_nc(self.stat_std_upper_file, False)
            surface_mean = self._stat_from_nc(self.stat_mean_std_file, True)
            surface_std = self._stat_from_nc(self.stat_std_std_file, True)
            # print(upper_mean.shape, upper_std.shape, surface_mean.shape, surface_std.shape)
        # with netCDF4.Dataset(path.join(self.root_dir,'stat_2007_mean_upper.nc')) as nc_upper_mean:
        #     upper_mean = torch.stack([torch.from_numpy(nc_upper_mean[var][:]) for var in self.upper_variables], dim=-1)
        #     upper_mean = torch.swapaxes(upper_mean.float().cpu(),0,1)
        # with netCDF4.Dataset(path.join(self.root_dir,'stat_2007_std_upper.nc')) as nc_upper_std:
        #     upper_std = torch.stack([torch.from_numpy(nc_upper_std[var][:]) for var in self.upper_variables], dim=-1)
        #     upper_std = torch.swapaxes(upper_std.float().cpu(),0,1)
        # with netCDF4.Dataset(path.join(self.root_dir,'stat_2007_mean_sfc.nc')) as nc_sfc_mean:
        #     surface_mean = torch.stack([torch.from_numpy(nc_sfc_mean[var][:]) for var in self.surface_variables], dim=-1)
        #     # when setting ingest_space_info=True, add `lon` and `lat` to the mean
        #     if self.ingest_space_info:
        #         shape = nc_sfc_mean[self.surface_variables[0]][:].shape
        #         lon = torch.zeros(shape)+130
        #         lat = torch.zeros(shape)+10
        #         surface_mean = torch.cat((surface_mean, lon.unsqueeze(-1), lat.unsqueeze(-1)), dim=-1)
        #     surface_mean = torch.swapaxes(surface_mean.float().cpu(),0,1)
        # with netCDF4.Dataset(path.join(self.root_dir,'stat_2007_std_sfc.nc')) as nc_sfc_std:
        #     surface_std = torch.stack([torch.from_numpy(nc_sfc_std[var][:]) for var in self.surface_variables], dim=-1)
        #     # when setting ingest_space_info=True, add `lon` and `lat` to the std
        #     if self.ingest_space_info:
        #         shape = nc_sfc_std[self.surface_variables[0]][:].shape
        #         lon = torch.zeros(shape)+12
        #         lat = torch.zeros(shape)+12
        #         surface_std = torch.cat((surface_std, lon.unsqueeze(-1), lat.unsqueeze(-1)), dim=-1)
        #     surface_std = torch.swapaxes(surface_std.float().cpu(),0,1)
        return upper_mean, upper_std, surface_mean, surface_std

    def _stack_nc(self, upper_nc: netCDF4.Dataset, surface_nc: netCDF4.Dataset) -> WeatherData:
        """
        Args:
            upper_nc (netCDF4.Dataset): Dataset of upper-air data.
            surface_nc (netCDF4.Dataset): Dataset of surface data.
        Returns:
            WeatherData: Tuple of upper-air data and surface data.
                         Upper-air data is of shape(pressure level, latitude, longitude, variable).
                         Surface data is of shape(latitude, longitude, variable).
        """
        upper_data = np.stack([np.squeeze(self._trim_var(upper_nc[v])) for v in self.upper_variables], axis=-1)
        upper_data = torch.FloatTensor(upper_data)
        surface_data = np.stack([np.squeeze(self._trim_var(surface_nc[v])) for v in self.surface_variables], axis=-1)
        # when setting ingest_space_info=True, add `longitude` and `latitude` to the data with values respectively
        if self.ingest_space_info:
            # get the `longitude` vector from the dataset
            lon = surface_nc['longitude'][:]
            # repeat the `lon` vector to match the shape of the data
            lon = np.tile(lon, (len(lon),1))
            lon = self._trim_var(lon)
            # get the `latitude` vector from the dataset
            lat = surface_nc['latitude'][:]
            # repeat the `lat` vector to match the shape of the data, and rearrange it to match the dataset
            lat = np.transpose(np.tile(lat,(len(lat),1)))
            lat = self._trim_var(lat)
            surface_data = np.concatenate((surface_data, np.expand_dims(lon,-1), np.expand_dims(lat,-1)), axis=-1)
        surface_data = torch.FloatTensor(surface_data)
        if self.standardize:
            upper_data = (upper_data - self.upper_mean) / self.upper_std
            surface_data = (surface_data - self.surface_mean) / self.surface_std
        return upper_data, surface_data

    def _index_to_path(self, index:int, target:bool=False) -> tuple[str, str]:
        """
        Args:
            date_hour (datetime): Date and hour to be converted to path.
        Returns:
            tuple[str, str]: Tuple of upper air path and surface path.
        """
        if self.combined_nc_input:
            nc_combined = self.file0_list_combined[index]
            if target:
                nc_combined = self.file1_list_combined[index]
            return nc_combined, nc_combined

        nc_upper = self.file0_list_upper[index]
        if target:
            nc_upper = self.file1_list_upper[index]

        nc_sfc = nc_upper.replace('_upper.','_sfc.')
        return nc_upper, nc_sfc

    def __len__(self) -> int:
        if self.combined_nc_input:
            return len(self.file0_list_combined)
        return len(self.file0_list_upper)

    def __getitem__(self, index: int) -> tuple[WeatherData, WeatherData] | tuple[WeatherData, WeatherData, Stat_T] | tuple[WeatherData, WeatherData, Stat_T, str]:
        """
        Args:
            index (int): Index of the hour to be retrieved.
        Returns:
            Return type depends on get_stat.
            tuple[WeatherData, WeatherData]: When get_stat is False, return tuple of input and target weather data.
                                             One WeatherData is composed of upper-air data and surface data.
                                             Upper-air data is of shape (pressure level, latitude, longitude, variable).
                                             Surface data is of shape (latitude, longitude, variable).
            tuple[WeatherData, WeatherData, Stat_T]: When get_stat is True, return tuple of input, target weather data
                                                     and (mean_upper, std_upper, mean_surface, std_surface).
        """
        stacked_input, stacked_target = None, None

        upper_path_input, surface_path_input = self._index_to_path(index)
        upper_path_target, surface_path_target = self._index_to_path(index,True)
        # print(f'{index}: {upper_path_input}, {surface_path_input}',flush=True)
        if self.plugin_additional_vars:
            with ntf(dir='/dev/shm',suffix='.nc') as tmp_nc:
                xds = combine_ncs_to_xarray(upper_path_input, surface_path_input)
                xds.to_netcdf(tmp_nc.name)
                tmp_nc.flush()
                combined_input = netCDF4.Dataset(tmp_nc.name, 'r')
                stacked_input = self._stack_nc(combined_input, combined_input)
                combined_input.close()
            with ntf(dir='/dev/shm',suffix='.nc') as tmp_nc:
                xds = combine_ncs_to_xarray(upper_path_target, surface_path_target)
                xds.to_netcdf(tmp_nc.name)
                tmp_nc.flush()
                combined_target = netCDF4.Dataset(tmp_nc.name, 'r')
                stacked_target = self._stack_nc(combined_target, combined_target)
                combined_target.close()
        else:
            upper_nc_input = (netCDF4.Dataset(upper_path_input, "r"))
            surface_input = (netCDF4.Dataset(surface_path_input, "r"))
            stacked_input = self._stack_nc(upper_nc_input, surface_input)
            upper_nc_input.close()
            surface_input.close()

            upper_nc_target = (netCDF4.Dataset(upper_path_target, "r"))
            surface_target = (netCDF4.Dataset(surface_path_target, "r"))
            stacked_target = self._stack_nc(upper_nc_target, surface_target)
            upper_nc_target.close()
            surface_target.close()

        if self.get_stat:
            if self.inferencing:
                case_file_name = '_'.join(path.basename(upper_path_input).split('_')[0:2])
                return stacked_input, stacked_target, (self.upper_mean, self.upper_std, self.surface_mean, self.surface_std), case_file_name
            else:
                return stacked_input, stacked_target, (self.upper_mean, self.upper_std, self.surface_mean, self.surface_std)
        else:
            return stacked_input, stacked_target

    def get_lon_lat_lev(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of longitude, latitude, and pressure level.
        """
        upper_nc_path, _ = self._index_to_path(self.start_year)
        upper_nc = netCDF4.Dataset(upper_nc_path, "r")
        lon = np.squeeze(upper_nc.variables["longitude"][:])
        lat = np.flip(np.squeeze(upper_nc.variables["latitude"][:]))
        lev = np.squeeze(upper_nc.variables["level"][:])
        return lon, lat, lev