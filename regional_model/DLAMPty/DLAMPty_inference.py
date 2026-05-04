import pandas as pd
import numpy as np
import yaml, sys, os
import xarray as xr
import onnxruntime as ort
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.data_processor import lonlat_uniformizer, recalc_additional_np,calc_additional_vars,to_xarray

class DLAMPty_model:
    def __init__(self, model_path, root_dir=os.path.dirname(__file__), device=None, cpu_num=10 ):
        self.model_path = model_path
        self.model_setting = yaml.safe_load(open(self.model_path))
        self.root_dir = root_dir
        self.device = device
        self.cpu_num = cpu_num
        self.uniformize_lonlat = True
        self.specify_resolution = 0.25
        
        self.onnx_path = os.path.join(self.root_dir,self.model_setting['onnx_path'])
        self.stat_mean_file = os.path.join(self.root_dir,self.model_setting['stat_mean_file'])
        self.stat_std_file = os.path.join(self.root_dir,self.model_setting['stat_std_file'])
        self.ingest_space_info = self.model_setting['ingest_space_info'] #ingest_space_info=True, add `lon` and `lat` to the std or mean
        self.upper_variables = self.model_setting['upper_vars']
        self.surface_variables = self.model_setting['surface_vars']
        self.pressure_levels = self.model_setting['pressure_levels']
        self.upper_units = self.model_setting['upper_units']
        self.surface_units = self.model_setting['surface_units']
        
    def _stat_from_nc(self, nc_file_path: str, want_sfc: bool) -> np.ndarray:
        """
        Args:
            nc_file_path (str): Path to the netCDF file.
            want_sfc (bool): Whether to return the mean or standard deviation of the surface data.

        Returns:
            np.ndarray: Mean or standard deviation of the upper-air data or surface data.
        """
        nc_path = nc_file_path
        with xr.open_dataset(nc_path) as input_nc:
            var_list = self.upper_variables if not want_sfc else self.surface_variables
            stat_data = np.stack([input_nc[var].values for var in var_list], axis=-1)
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
                lon = np.expand_dims(np.zeros(shape)+c_lon, axis=-1) 
                lat = np.expand_dims(np.zeros(shape)+c_lat, axis=-1) 
                stat_data = np.concatenate((stat_data, lon, lat), axis=-1)
            stat_data = stat_data = np.swapaxes(stat_data, 0, 1)
        return stat_data
    
    def load_statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of mean and std of upper-air data and surface data.
        """
        self.upper_mean = self._stat_from_nc(self.stat_mean_file, False)
        self.upper_std = self._stat_from_nc(self.stat_std_file, False)
        self.surface_mean = self._stat_from_nc(self.stat_mean_file, True)
        self.surface_std = self._stat_from_nc(self.stat_std_file, True)

    def load_model(self):        
        cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
        ort_providers = [
            ("CUDAExecutionProvider", cuda_provider_options),
            "CPUExecutionProvider",
        ]
        if self.device=='cpu' or not os.path.exists("/proc/driver/nvidia/version"):
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.cpu_num 
            ort_providers.pop(0)
            self.device = 'cpu'
            model = ort.InferenceSession(self.onnx_path, sess_options=session_options, providers=ort_providers)
        else:
            model = ort.InferenceSession(self.onnx_path, providers=ort_providers)
        print(f"inference with {ort_providers}")
        return model
    
    def initialize(self):
        self.model = self.load_model() 
        self.load_statistics()
        print("Model and weights are loaded.")
        
    def normalize(self, upper_data, surface_data, reverse=False):
        if reverse:
            new_upper = upper_data * self.upper_std + self.upper_mean
            new_surface = surface_data * self.surface_std + self.surface_mean
        else:
            new_upper = (upper_data - self.upper_mean) / self.upper_std 
            new_surface = (surface_data - self.surface_mean) / self.surface_std
        return new_upper, new_surface
        
    def predict_one_step(self, input_upper, input_surface):
        
        input_upper,input_surface = self.normalize(input_upper,input_surface)

        input_upper = np.expand_dims(input_upper, axis=0)
        input_surface = np.expand_dims(input_surface, axis=0)
        
        ort_inputs = {
            "input_upper": input_upper.astype(np.float32),
            "input_surface": input_surface.astype(np.float32)
        }
        
        # Run inference
        ort_outputs = self.model.run(None,ort_inputs)
        output_upper = ort_outputs[0].squeeze()
        output_surface = ort_outputs[1].squeeze()

        # reverse back
        output_upper, output_surface = self.normalize(output_upper,output_surface,reverse=True)
        # uniform lat lon
        if self.uniformize_lonlat:
            lon, lat = lonlat_uniformizer(
            output_surface[:, :, -2],
            output_surface[:, :, -1],
            self.uniformize_lonlat,
            self.specify_resolution,
        )

        (output_surface[:, :, -2], output_surface[:, :, -1]) = np.meshgrid(lon, lat)

        return output_upper, output_surface
    
    def changing_additional_information(self, input_upper, input_surface, timestep):
        additionals = recalc_additional_np(
                input_upper, input_surface, timestep,
                self.upper_variables, self.surface_variables, self.upper_units, self.surface_units
            )

        for v in self.surface_variables:
            if v in additionals:
                i = self.surface_variables.index(v)
                input_surface[:, :, i] = additionals[v]

        for v in self.upper_variables:
            if v in additionals:
                i = self.upper_variables.index(v)
                input_upper[:, :, :, i] = additionals[v]
        
        return input_upper, input_surface
    
    def IC_from_xarray_to_npy(self, IC_dataset:xr.Dataset, additional_vars=False):
        if not additional_vars:
            print('It needs to calc additional vars.')
            IC_dataset = calc_additional_vars(IC_dataset, True)
        input_upper = np.stack([IC_dataset[var].values for var in self.upper_variables], axis=-1).squeeze()
        input_surface = np.stack([IC_dataset[var].values for var in self.surface_variables], axis=-1).squeeze()
        lon, lat = np.meshgrid(IC_dataset.longitude, IC_dataset.latitude)
        input_surface = np.concatenate((input_surface, np.stack([lon, lat],axis=-1)), axis=-1)
        return input_upper, input_surface

    def data_to_xarray(self, upper_data, surface_data, timestep):
        DLAMPty_xr = to_xarray(upper_data, surface_data, self.upper_variables, self.surface_variables, self.upper_units, self.surface_units, self.pressure_levels)
        DLAMPty_xr = DLAMPty_xr.expand_dims(time=[pd.to_datetime(timestep)])
        return DLAMPty_xr