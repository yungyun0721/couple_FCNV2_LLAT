import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import argparse

from global_model.FCNV2.FCNV2_inference import FCNV2_model
from regional_model.DLAMPty.DLAMPty_inference import DLAMPty_model
from interaction_tools.FCNV2_DLAMPty_interaction import transfer_FCNV2_DLAMPty_with_radius

FCNV2_info = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",
                "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
                "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925","v1000",
                "z50",  "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925","z1000",
                "t50",  "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925","t1000",
                "r50",  "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

LLAT_info = {
        "model_title": "v57 5d (SG, ra)",
        "upper_vars": ['u', 'v', 't', 'q', 'z', 'w'],
        "upper_units": ["m/s", "m/s", "K", "kg kg**-1", "m**2 s**-2", "Pa s**-1"],
        "surface_vars": ['u10', 'v10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf', 'sst_filled', 'f', 'solar', 'hgt', 'landmask', 'diurnal_sin', 'diurnal_cos', 'doy_sin', 'doy_cos'],
        "surface_units": ["m/s", "m/s", "K", "K", "Pa", "Pa", "kg m**-2", "m", 'W m**-2', 'K', '1/s', 'W m**-2', 'm', '1', '1', '1', '1', '1'],
        "pressure_levels" : [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        'coastline_color': 'darkslategray',
    }

def main(FCNV2_IC_path, LLAT_IC_path, IC_time, save_folder, fore_hour=72,
         FCNV2_weight="global_model/FCNV2/weight",FCNV2_device='cuda',
         LLAT_yaml = "regional_model/DLAMPty/onnx/v57_5d.yaml", LLAT_device='cpu'):    
    
    # replace vars
    upper_vars = ['u', 'v', 't', 'q', 'z']    
    surface_vars = ['u10', 'v10', 't2m', 'msl', 'sp', 'tcwv']

    # save FCNV2 small domain
    initial_time = datetime.datetime.strptime(str(IC_time), "%Y%m%d%H")
    lat = np.flip(np.linspace(-90,90,721))
    lon = np.linspace(0,359.75,1440)

    lat_min  = np.argwhere(lat==-10)[0][0]
    lat_max  = np.argwhere(lat==80)[0][0]
    lon_min  = np.argwhere(lon==80)[0][0]
    lon_max  = np.argwhere(lon==180)[0][0]
    
    # initialize model
    print('initialize model ...')
    print(f'FCNV2 weight: {FCNV2_weight}, device: {FCNV2_device}')
    FCNV2 = FCNV2_model(FCNV2_weight, device=FCNV2_device)
    FCNV2.initialize()
    print(f'LLAT.yaml: {LLAT_yaml}, device: {LLAT_device}')
    LLAT = DLAMPty_model(LLAT_yaml,device=LLAT_device)
    LLAT.initialize()


    # save folder    
    os.makedirs(save_folder, exist_ok=True)
    print(f'save folder: {save_folder}')
    FCNV2_save_path = os.path.join(save_folder,'FCNV2')
    os.makedirs(FCNV2_save_path, exist_ok=True)
    LLAT_save_path = os.path.join(save_folder,'LLAT')
    os.makedirs(LLAT_save_path, exist_ok=True)

    with open(f"log.txt", "a") as f:
        print('\n',file=f)
        print(f'start time IC {initial_time}-------',file=f)
        print(f'start time: {datetime.datetime.now()}',file=f)

    # building save folder
    # load and save IC_data
    print('load IC data and save ...')
    print(f'FCNV2 IC path: {FCNV2_IC_path}')
    print(f'LLAT IC path: {LLAT_IC_path}')
    FCNV2_input = np.load(FCNV2_IC_path)
    LLAT_IC = xr.open_dataset(LLAT_IC_path)
    LLAT_input_upper,LLAT_input_surface = LLAT.IC_from_xarray_to_npy(LLAT_IC)

    np.save(os.path.join(FCNV2_save_path, f"output_weather_{0:0>3}h"),FCNV2_input[:,lat_max:lat_min,lon_min:lon_max])
    np.save(os.path.join(LLAT_save_path,  f"output_upper_{0:0>3}h"),LLAT_input_upper)
    np.save(os.path.join(LLAT_save_path,  f"output_sfc_{0:0>3}h"),LLAT_input_surface)

    for fore_i in range(1, np.int_(fore_hour/6)+1):
        
        # running FCNV2 forecast
        FCNV2_output = FCNV2.predict_one_step(FCNV2_input)
        np.save(os.path.join(FCNV2_save_path, f"output_weather_{fore_i*6:0>3}h"),FCNV2_output[:,lat_max:lat_min,lon_min:lon_max])
        FCNV2_input = FCNV2_output
        
        # running LLAT forecast
        # 3 hr
        target_time = initial_time+ datetime.timedelta(hours=(fore_i*6-3))
        LLAT_output_upper, LLAT_output_surface = LLAT.predict_one_step(LLAT_input_upper,LLAT_input_surface)
        np.save(os.path.join(LLAT_save_path, f"output_upper_{(fore_i*6-3):0>3}h"),LLAT_output_upper)
        np.save(os.path.join(LLAT_save_path, f"output_sfc_{(fore_i*6-3):0>3}h"),LLAT_output_surface)
        LLAT_input_upper, LLAT_input_surface = LLAT.changing_additional_information(LLAT_output_upper, LLAT_output_surface,target_time)
        # 6 hr
        target_time = initial_time+ datetime.timedelta(hours=(fore_i*6))
        LLAT_output_upper, LLAT_output_surface = LLAT.predict_one_step(LLAT_input_upper,LLAT_input_surface)
        np.save(os.path.join(LLAT_save_path, f"output_upper_{(fore_i*6):0>3}h"),LLAT_output_upper)
        np.save(os.path.join(LLAT_save_path, f"output_sfc_{(fore_i*6):0>3}h"),LLAT_output_surface)
        LLAT_input_upper, LLAT_input_surface = LLAT.changing_additional_information(LLAT_output_upper, LLAT_output_surface,target_time)
        

        #load initial bdy
        LLAT_bdy_path = LLAT_IC_path
        # DLAMPty_bdy_path = f'/wk2/yungyun/FCNV2_TC_JTWC/{TC_ID}/ERA5/for_DLAMPty/{TC_ID}_{target_time.strftime("%Y%m%d%H")}_combined.nc'
        # DLAMPty_bdy_path = f'/wk2/pc/AI_models/RegionalCouple_AI/ERA5_DLANPty/{TC_ID}/{TC_ID}_{target_time.strftime("%Y%m%d%H")}_combined.nc'
        LLAT_bdy = xr.open_dataset(LLAT_bdy_path)
        LLAT_bdy_upper,LLAT_bdy_surface = LLAT.IC_from_xarray_to_npy(LLAT_bdy)

        # change bdy
        for var in upper_vars:
            LLAT_input_upper[:,:8,:,LLAT.model_setting['upper_vars'].index(var)] = LLAT_bdy_upper[:,:8,:,LLAT.model_setting['upper_vars'].index(var)] 
            LLAT_input_upper[:,-8:,:,LLAT.model_setting['upper_vars'].index(var)] = LLAT_bdy_upper[:,-8:,:,LLAT.model_setting['upper_vars'].index(var)] 
            LLAT_input_upper[:,:,:8,LLAT.model_setting['upper_vars'].index(var)] = LLAT_bdy_upper[:,:,:8,LLAT.model_setting['upper_vars'].index(var)] 
            LLAT_input_upper[:,:,-8:,LLAT.model_setting['upper_vars'].index(var)] = LLAT_bdy_upper[:,:,-8:,LLAT.model_setting['upper_vars'].index(var)]
        
        for var in surface_vars:            
            LLAT_input_surface[:8,:,LLAT.model_setting['surface_vars'].index(var)] = LLAT_bdy_surface[:8,:,LLAT.model_setting['surface_vars'].index(var)] 
            LLAT_input_surface[-8:,:,LLAT.model_setting['surface_vars'].index(var)] = LLAT_bdy_surface[-8:,:,LLAT.model_setting['surface_vars'].index(var)] 
            LLAT_input_surface[:,:8,LLAT.model_setting['surface_vars'].index(var)] = LLAT_bdy_surface[:,:8,LLAT.model_setting['surface_vars'].index(var)] 
            LLAT_input_surface[:,-8:,LLAT.model_setting['surface_vars'].index(var)] = LLAT_bdy_surface[:,-8:,LLAT.model_setting['surface_vars'].index(var)] 
            
        if np.mod(fore_i,4)==0:
            with open(f"log.txt", "a") as f:
                print(f'finish IC {initial_time.strftime("%Y%m%d%H")} : forecast {(fore_i*6):0>3} hr {target_time.strftime("%Y%m%d%H")}',file=f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments to get YAML config file.
    parser.add_argument('--TC_ID',  required=True,
                        help='TC_ID for TC_list (.csv)')
    args = parser.parse_args()  
    
    main(str(args.TC_ID))     

