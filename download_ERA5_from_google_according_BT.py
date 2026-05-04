import numpy as np
import datetime, os, gcsfs
import xarray as xr
import argparse
import pandas as pd
gcs = gcsfs.GCSFileSystem(token='anon')
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xr.open_zarr(gcs.get_mapper(era5_path), chunks=None)


def q_to_rh(q, T_K, p):
    """
    q: 比濕 [kg/kg]
    T_K: 溫度 [K]
    p: 氣壓 [hPa]
    回傳 RH [%]
    """
    p = (np.array(p)[:,np.newaxis]*np.ones([1,q.shape[1]]))[:,:,np.newaxis]*np.ones([1,q.shape[2]])
    T_C = T_K - 273.15  # 轉為攝氏
    w = q / (1 - q)
    e = (w * p) / (0.622 + w)
    es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    rh = (e / es) * 100
    return rh


# IC_time = "2026041400"
# TC_center_lat = 14.2
# TC_center_lon = 146.6
# save_folder = "input_data"
def main(BT_track_path: str, save_folder: str):
    os.makedirs(save_folder, exist_ok=True)

    TC_track = pd.read_csv(BT_track_path)
    TC_track['datetime'] = pd.to_datetime(TC_track[['Year', 'Month', 'Day', 'Hour']])

    # for hour_i in range(len(TC_track)):   
    for hour_i in range(1):           
        IC_time = TC_track['datetime'].iloc[hour_i]
        TC_center_lat = TC_track['Lat.'].iloc[hour_i]
        TC_center_lon = TC_track['Long.'].iloc[hour_i]

        # for getting LLAT.ty variable names
        upper_variable = ['u_component_of_wind','v_component_of_wind', 'temperature','specific_humidity','geopotential','vertical_velocity']
        surface_variable = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_dewpoint_temperature', 
                            'mean_sea_level_pressure','surface_pressure','total_column_water_vapour','total_precipitation',
                            'mean_top_net_long_wave_radiation_flux','sea_surface_temperature']
        target_lev = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
        era5_surface_original = full_era5[surface_variable].sel(time=IC_time.strftime("%Y-%m-%d %H"))
        era5_upper_original = full_era5[upper_variable].sel(time=IC_time.strftime("%Y-%m-%d %H"),level=target_lev)
        ############# prepare LLAT.ty input data
        TC_lat_index = np.arange(np.int_((90-TC_center_lat)/0.25)-40,np.int_((90-TC_center_lat)/0.25)+41)
        TC_lon_index = np.arange(np.int_((TC_center_lon)/0.25)-40,np.int_((TC_center_lon)/0.25)+41)
        new_upper = era5_upper_original.isel(latitude=TC_lat_index, longitude=TC_lon_index).expand_dims({"time": [IC_time]})
        new_surface= era5_surface_original.isel(latitude=TC_lat_index, longitude=TC_lon_index).expand_dims({"time": [IC_time]})

        new_upper = new_upper.rename({
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "temperature": "t",
            "specific_humidity": "q",
            "geopotential": "z",
            "vertical_velocity": "w"
        })

        new_surface = new_surface.rename({
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "2m_temperature": "t2m",
            "2m_dewpoint_temperature": "d2m",
            "mean_sea_level_pressure": "msl",
            "surface_pressure": "sp",
            "total_column_water_vapour": "tcwv",
            "total_precipitation": "tp",
            "mean_top_net_long_wave_radiation_flux": "mtnlwrf",
            "sea_surface_temperature": "sst",
        })

        total_LLAT_input = xr.merge([new_upper, new_surface])
        total_LLAT_input.to_netcdf(f'{save_folder}/{IC_time.strftime("%Y%m%d%H")}_combined.nc')
        print(f'finish LLAT IC:{IC_time.strftime("%Y%m%d%H")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--BT_track_path", required=True, help="the path from BT track")
    parser.add_argument("--save-folder", default="ERA5_data", help="folder to save input data")
    args = parser.parse_args()
    main(args.BT_track_path, args.save_folder) 