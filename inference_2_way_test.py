import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import argparse

from global_model.FCNV2.FCNV2_inference import FCNV2_model
from regional_model.DLAMPty.DLAMPty_inference import DLAMPty_model
from interaction_tools.FCNV2_DLAMPty_interaction import transfer_FCNV2_DLAMPty_with_radius

def main(FCNV2_IC_path, LLAT_IC_path, IC_time, save_folder, fore_hour=72,
         FCNV2_weight="/wk2/yungyun/code_space/FCNV2_test/weight",FCNV2_device='cuda',
         LLAT_yaml = "regional_model/DLAMPty/onnx/v57_5d.yaml", LLAT_device='cpu'):    
    
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
        
        # 2-way interaction
        FCNV2_input, LLAT_input_upper, LLAT_input_surface = transfer_FCNV2_DLAMPty_with_radius(
            FCNV2_input, 
            LLAT_input_upper, 
            LLAT_input_surface,
            LLAT.model_setting,
            radius=7.5)
            

        if np.mod(fore_i,4)==0:
            with open(f"log.txt", "a") as f:
                print(f'finish IC {initial_time.strftime("%Y%m%d%H")} : forecast {(fore_i*6):0>3} hr {target_time.strftime("%Y%m%d%H")}',file=f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments to get YAML config file.
    parser.add_argument('--FCNV2_IC_path',  required=True, help='Path to FCNV2 IC data')
    parser.add_argument('--LLAT_IC_path',  required=True, help='Path to LLAT IC data')
    parser.add_argument('--IC_time',  required=True, help='Initial time for IC data (ex: 2026041300)')
    parser.add_argument('--save_folder',  required=True, help='Folder to save results')
    parser.add_argument('--fore_hour',  default=72, help='Forecast hours')
    args = parser.parse_args()  
    
    main(args.FCNV2_IC_path, args.LLAT_IC_path, args.IC_time, args.save_folder, fore_hour=args.fore_hour)     

