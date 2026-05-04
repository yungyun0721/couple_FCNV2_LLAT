import numpy as np
import xarray as xr
from utils.data_processor import to_xarray
import datetime
from pathlib import Path

# TC_ID = '202405W'
# start_time = '2024072000'
# step = 2

# # DLAMPty data
# DLAMPty_path = f'/wk2/yungyun/FCNV2_TC/{TC_ID}/DLAMP_test/out_v57_5d_forecast_from_{TC_ID}_{start_time}_combined.nc_uni_ra/_{start_time}/' 
# save_DLAMPty_path=f'/wk2/yungyun/FCNV2_TC/{TC_ID}/2_way_interact/{start_time}/medium/'

# FCNV2_path = f'/wk2/yungyun/FCNV2_TC/{TC_ID}/FCNV2_forecast/{start_time}/'
# save_FCNV2_path=f'/wk2/yungyun/FCNV2_TC/{TC_ID}/2_way_interact/{start_time}/medium/'

def q_to_rh(q, T_K, p):
    """
    q: 比濕 [kg/kg]
    T_K: 溫度 [K]
    p: 氣壓 [hPa]
    回傳 RH [%]
    """
    T_C = T_K - 273.15  # 轉為攝氏
    w = q / (1 - q)
    e = (w * p) / (0.622 + w)
    es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    rh = (e / es) * 100
    return rh

def rh_to_q(rh, T_K, p):
    """
    rh: 相對濕度 [%]
    T_K: 溫度 [K]
    p: 氣壓 [hPa]
    回傳 q [kg/kg]
    """
    T_C = T_K - 273.15
    es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    e = (rh / 100) * es
    w = 0.622 * e / (p - e)
    q = w / (1 + w)
    return q

def change_surface_bdy(DLAMPty, FCNV2):
    DLAMPty[:8,:] = FCNV2[:8,:]
    DLAMPty[-8:,:] = FCNV2[-8:,:]
    DLAMPty[:,:8] = FCNV2[:,:8]
    DLAMPty[:,-8:] = FCNV2[:,-8:]
    return DLAMPty

def change_upper_bdy(DLAMPty, FCNV2):
    DLAMPty[:,:8,:] = FCNV2[:, :8,:]
    DLAMPty[:,-8:,:] = FCNV2[:, -8:,:]
    DLAMPty[:,:,:8] = FCNV2[:, :,:8]
    DLAMPty[:,:,-8:] = FCNV2[:, :,-8:]
    return DLAMPty

def transfer_FCNV2_DLAMPty(FCNV2_path, DLAMPty_path, medium_save_path, file_time, step=2):
    file_datetime = datetime.datetime.strptime(file_time, '%Y%m%d%H')
    target_time = (file_datetime + datetime.timedelta(hours=step*3)).strftime('%Y%m%d%H')

    info = {
            "model_title": "v57 5d (SG, ra)",
            "upper_vars": ['u', 'v', 't', 'q', 'z', 'w'],
            "upper_units": ["m/s", "m/s", "K", "kg kg**-1", "m**2 s**-2", "Pa s**-1"],
            "surface_vars": ['u10', 'v10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf', 'sst_filled', 'f', 'solar', 'hgt', 'landmask', 'diurnal_sin', 'diurnal_cos', 'doy_sin', 'doy_cos'],
            "surface_units": ["m/s", "m/s", "K", "K", "Pa", "Pa", "kg m**-2", "m", 'W m**-2', 'K', '1/s', 'W m**-2', 'm', '1', '1', '1', '1', '1'],
            "pressure_levels" : [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            'coastline_color': 'darkslategray',
        }

    DLAMPty_upper = np.load(Path(DLAMPty_path, f"output_upper_{step:0>3}.npy"))
    DLAMPty_sfc = np.load(Path(DLAMPty_path, f"output_sfc_{step:0>3}.npy"))
    DLAMPty_xr = to_xarray(DLAMPty_upper, DLAMPty_sfc, info['upper_vars'], info['surface_vars'], info['upper_units'], info['surface_units'], info['pressure_levels'])
    pressure_matrix = (np.array(DLAMPty_xr.level)[:,np.newaxis]*np.ones([1,81]))[:,:,np.newaxis]*np.ones([1,81])
    
    print(f'DLAMPty from {DLAMPty_path}output_upper_{step:0>3}.npy')
    
    #FCNV2 data
    ordering = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",
                "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
                "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850"
                "v925","v1000",  "z50", "z100", "z150", "z200", "z250", "z300",\
                "z400", "z500", "z600", "z700", "z850", "z925","z1000",  "t50",\
                "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600",\
                "t700", "t850", "t925","t1000",  "r50", "r100", "r150", "r200",\
                "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

    FCNV2_data_original = np.load(f'{FCNV2_path}output_weather_{step*3}h.npy')
    
    print(f'FCNV2 from {FCNV2_path}output_weather_{step*3}h.npy')
    
    
    # find TC center
    TC_center_lat = np.round(DLAMPty_xr.latitude[40].item(),2).astype(np.float32)
    TC_center_lon = np.round(DLAMPty_xr.longitude[40].item(),2).astype(np.float32)
    # find index
    lat_index = np.arange(np.int_((90-TC_center_lat)/0.25)-40,np.int_((90-TC_center_lat)/0.25)+41)
    lon_index = np.arange(np.int_((TC_center_lon)/0.25)-40,np.int_((TC_center_lon)/0.25)+41)

    FCNV2_data = FCNV2_data_original.copy()

    # change FCNV2
    #surface
    FCNV2_data[0, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.u10)
    FCNV2_data[1, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.v10)
    FCNV2_data[4, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.t2m)
    FCNV2_data[5, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.sp)
    FCNV2_data[6, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.msl)
    FCNV2_data[7, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.tcwv)
    #upper
    FCNV2_data[ 8:21, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.u)
    FCNV2_data[21:34, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.v)
    FCNV2_data[34:47, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.z)
    FCNV2_data[47:60, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_xr.t)
    DLAMPty_RH = q_to_rh(DLAMPty_xr.q,DLAMPty_xr.t,pressure_matrix)
    FCNV2_data[60:, lat_index[0]:lat_index[-1]+1, lon_index[0]:lon_index[-1]+1] = np.array(DLAMPty_RH)
    # with open(f"log.txt", "a") as f:
    #     print(f"----tcwv check--------", file=f)
    #     print(f"tcwv diff{np.min(FCNV2_data[7, lat_index, :][:,lon_index]-np.array(DLAMPty_xr.tcwv))}",file=f)
    #     print(f'FCNV2 originial tcwv check {FCNV2_data_original[7, np.int_((90-TC_center_lat)/0.25), np.int_((TC_center_lon)/0.25)]}',file=f)
    #     print(f'FCNV2 tcwv check {FCNV2_data[7, np.int_((90-TC_center_lat)/0.25), np.int_((TC_center_lon)/0.25)]}',file=f)
    #     print(f'DLAMPty tcwv check {np.array(DLAMPty_xr.tcwv)[40,40]}',file=f)
    np.save(f'{medium_save_path}/FCNV2_{target_time}.npy',FCNV2_data.astype(np.float32))

    #change surface DLAMPty
    DLAMPty_xr['u10'] = change_surface_bdy(DLAMPty_xr.u10,FCNV2_data_original[0, lat_index, :][:,lon_index])
    DLAMPty_xr['v10'] = change_surface_bdy(DLAMPty_xr.v10,FCNV2_data_original[1, lat_index, :][:,lon_index])
    DLAMPty_xr['t2m'] = change_surface_bdy(DLAMPty_xr.t2m,FCNV2_data_original[4, lat_index, :][:,lon_index])
    DLAMPty_xr['sp'] = change_surface_bdy(DLAMPty_xr.sp,FCNV2_data_original[5, lat_index, :][:,lon_index])
    DLAMPty_xr['msl'] = change_surface_bdy(DLAMPty_xr.msl,FCNV2_data_original[6, lat_index, :][:,lon_index])
    DLAMPty_xr['tcwv'] = change_surface_bdy(DLAMPty_xr.tcwv,FCNV2_data_original[7, lat_index, :][:,lon_index])

    #change upper DLAMPty
    DLAMPty_xr['u'] = change_upper_bdy(DLAMPty_xr.u, FCNV2_data[ 8:21, lat_index, :][:, :,lon_index])
    DLAMPty_xr['v'] = change_upper_bdy(DLAMPty_xr.v, FCNV2_data[21:34, lat_index, :][:, :,lon_index])
    DLAMPty_xr['z'] = change_upper_bdy(DLAMPty_xr.z, FCNV2_data[34:47, lat_index, :][:, :,lon_index])
    DLAMPty_xr['t'] = change_upper_bdy(DLAMPty_xr.t, FCNV2_data[47:60, lat_index, :][:, :,lon_index])
    FCNV2_q = rh_to_q(FCNV2_data[60:73, lat_index, :][:, :,lon_index],FCNV2_data[47:60, lat_index, :][:, :,lon_index],pressure_matrix)
    DLAMPty_xr['q'] = change_upper_bdy(DLAMPty_xr.q, FCNV2_q)

    DLAMPty_xr = DLAMPty_xr.assign_coords(longitude = np.arange(-40,41)*0.25+TC_center_lon)
    DLAMPty_xr = DLAMPty_xr.assign_coords(latitude = np.flip(np.arange(-40,41)*0.25+TC_center_lat))
    DLAMPty_xr = DLAMPty_xr.drop_vars(["f", "solar", "hgt", "landmask", "diurnal_sin", "diurnal_cos", "doy_sin", "doy_cos","ws10","vort10"])
    DLAMPty_xr = DLAMPty_xr.drop_vars(["raw_lon", "raw_lat", "streamplot_lon", "streamplot_lat", "ws", "vort", "dewpoint", "theta_e"])
    DLAMPty_xr = DLAMPty_xr.expand_dims({"time": [datetime.datetime.strptime(target_time, "%Y%m%d%H")]})
    DLAMPty_xr.to_netcdf(f'{medium_save_path}/DLAMPty_{target_time}_combined.nc')
    return


def transfer_FCNV2_DLAMPty_with_radius(FCNV2_path, DLAMPty_path, medium_save_path, file_time, step=2, radius=7.5):
    
    xx, yy = np.meshgrid(np.arange(81),np.arange(81))
    dis_grid = np.sqrt(((xx-40)*0.25)**2+((yy-40)*0.25)**2)
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    dis_grid = dis_grid.reshape([-1])
    grid_mask = np.ones(xx.shape) 
    grid_mask[dis_grid>radius] = np.nan
    mask_xx = xx[~np.isnan(grid_mask)]
    mask_yy = yy[~np.isnan(grid_mask)]

    file_datetime = datetime.datetime.strptime(file_time, '%Y%m%d%H')
    target_time = (file_datetime + datetime.timedelta(hours=step*3)).strftime('%Y%m%d%H')

    info = {
            "model_title": "v57 5d (SG, ra)",
            "upper_vars": ['u', 'v', 't', 'q', 'z', 'w'],
            "upper_units": ["m/s", "m/s", "K", "kg kg**-1", "m**2 s**-2", "Pa s**-1"],
            "surface_vars": ['u10', 'v10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf', 'sst_filled', 'f', 'solar', 'hgt', 'landmask', 'diurnal_sin', 'diurnal_cos', 'doy_sin', 'doy_cos'],
            "surface_units": ["m/s", "m/s", "K", "K", "Pa", "Pa", "kg m**-2", "m", 'W m**-2', 'K', '1/s', 'W m**-2', 'm', '1', '1', '1', '1', '1'],
            "pressure_levels" : [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            'coastline_color': 'darkslategray',
        }

    DLAMPty_upper = np.load(Path(DLAMPty_path, f"output_upper_{step:0>3}.npy"))
    DLAMPty_sfc = np.load(Path(DLAMPty_path, f"output_sfc_{step:0>3}.npy"))
    DLAMPty_xr = to_xarray(DLAMPty_upper, DLAMPty_sfc, info['upper_vars'], info['surface_vars'], info['upper_units'], info['surface_units'], info['pressure_levels'])
    pressure_matrix = (np.array(DLAMPty_xr.level)[:,np.newaxis]*np.ones([1,81]))[:,:,np.newaxis]*np.ones([1,81])
    
    print(f'DLAMPty from {DLAMPty_path}output_upper_{step:0>3}.npy')
    
    #FCNV2 data
    ordering = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",
                "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
                "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850"
                "v925","v1000",  "z50", "z100", "z150", "z200", "z250", "z300",\
                "z400", "z500", "z600", "z700", "z850", "z925","z1000",  "t50",\
                "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600",\
                "t700", "t850", "t925","t1000",  "r50", "r100", "r150", "r200",\
                "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

    FCNV2_data_original = np.load(f'{FCNV2_path}output_weather_{step*3}h.npy')
    
    print(f'FCNV2 from {FCNV2_path}output_weather_{step*3}h.npy')
    
    
    # find TC center
    TC_center_lat = np.round(DLAMPty_xr.latitude[40].item(),2).astype(np.float32)
    TC_center_lon = np.round(DLAMPty_xr.longitude[40].item(),2).astype(np.float32)
    # find index
    lat_change_FCNV2_index = mask_xx + np.int_((90-TC_center_lat)/0.25)-40
    lon_change_FCNV2_index = mask_yy + np.int_((TC_center_lon)/0.25)-40

    lat_index = np.arange(np.int_((90-TC_center_lat)/0.25)-40,np.int_((90-TC_center_lat)/0.25)+41)
    lon_index = np.arange(np.int_((TC_center_lon)/0.25)-40,np.int_((TC_center_lon)/0.25)+41)

    FCNV2_data = FCNV2_data_original.copy()

    # change FCNV2
    #surface
    FCNV2_data[0, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.u10.values[mask_xx,mask_yy])
    FCNV2_data[1, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.v10.values[mask_xx,mask_yy])
    FCNV2_data[4, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.t2m.values[mask_xx,mask_yy])
    FCNV2_data[5, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.sp.values[mask_xx,mask_yy])
    FCNV2_data[6, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.msl.values[mask_xx,mask_yy])
    FCNV2_data[7, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.tcwv.values[mask_xx,mask_yy])
    #upper
    FCNV2_data[ 8:21, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.u.values[:,mask_xx,mask_yy])
    FCNV2_data[21:34, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.v.values[:,mask_xx,mask_yy])
    FCNV2_data[34:47, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.z.values[:,mask_xx,mask_yy])
    FCNV2_data[47:60, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_xr.t.values[:,mask_xx,mask_yy])
    DLAMPty_RH = q_to_rh(DLAMPty_xr.q,DLAMPty_xr.t,pressure_matrix)
    FCNV2_data[60:, lat_change_FCNV2_index, lon_change_FCNV2_index] = np.array(DLAMPty_RH.values[:,mask_xx,mask_yy])
    # with open(f"log.txt", "a") as f:
    #     print(f"----tcwv check--------", file=f)
    #     print(f"tcwv diff{np.min(FCNV2_data[7, lat_index, :][:,lon_index]-np.array(DLAMPty_xr.tcwv))}",file=f)
    #     print(f'FCNV2 originial tcwv check {FCNV2_data_original[7, np.int_((90-TC_center_lat)/0.25), np.int_((TC_center_lon)/0.25)]}',file=f)
    #     print(f'FCNV2 tcwv check {FCNV2_data[7, np.int_((90-TC_center_lat)/0.25), np.int_((TC_center_lon)/0.25)]}',file=f)
    #     print(f'DLAMPty tcwv check {np.array(DLAMPty_xr.tcwv)[40,40]}',file=f)
    np.save(f'{medium_save_path}/FCNV2_{target_time}.npy',FCNV2_data.astype(np.float32))

    #change surface DLAMPty
    DLAMPty_xr['u10'] = change_surface_bdy(DLAMPty_xr.u10,FCNV2_data_original[0, lat_index, :][:,lon_index])
    DLAMPty_xr['v10'] = change_surface_bdy(DLAMPty_xr.v10,FCNV2_data_original[1, lat_index, :][:,lon_index])
    DLAMPty_xr['t2m'] = change_surface_bdy(DLAMPty_xr.t2m,FCNV2_data_original[4, lat_index, :][:,lon_index])
    DLAMPty_xr['sp'] = change_surface_bdy(DLAMPty_xr.sp,FCNV2_data_original[5, lat_index, :][:,lon_index])
    DLAMPty_xr['msl'] = change_surface_bdy(DLAMPty_xr.msl,FCNV2_data_original[6, lat_index, :][:,lon_index])
    DLAMPty_xr['tcwv'] = change_surface_bdy(DLAMPty_xr.tcwv,FCNV2_data_original[7, lat_index, :][:,lon_index])

    #change upper DLAMPty
    DLAMPty_xr['u'] = change_upper_bdy(DLAMPty_xr.u, FCNV2_data[ 8:21, lat_index, :][:, :,lon_index])
    DLAMPty_xr['v'] = change_upper_bdy(DLAMPty_xr.v, FCNV2_data[21:34, lat_index, :][:, :,lon_index])
    DLAMPty_xr['z'] = change_upper_bdy(DLAMPty_xr.z, FCNV2_data[34:47, lat_index, :][:, :,lon_index])
    DLAMPty_xr['t'] = change_upper_bdy(DLAMPty_xr.t, FCNV2_data[47:60, lat_index, :][:, :,lon_index])
    FCNV2_q = rh_to_q(FCNV2_data[60:73, lat_index, :][:, :,lon_index],FCNV2_data[47:60, lat_index, :][:, :,lon_index],pressure_matrix)
    DLAMPty_xr['q'] = change_upper_bdy(DLAMPty_xr.q, FCNV2_q)

    DLAMPty_xr = DLAMPty_xr.assign_coords(longitude = np.arange(-40,41)*0.25+TC_center_lon)
    DLAMPty_xr = DLAMPty_xr.assign_coords(latitude = np.flip(np.arange(-40,41)*0.25+TC_center_lat))
    DLAMPty_xr = DLAMPty_xr.drop_vars(["f", "solar", "hgt", "landmask", "diurnal_sin", "diurnal_cos", "doy_sin", "doy_cos","ws10","vort10"])
    DLAMPty_xr = DLAMPty_xr.drop_vars(["raw_lon", "raw_lat", "streamplot_lon", "streamplot_lat", "ws", "vort", "dewpoint", "theta_e"])
    DLAMPty_xr = DLAMPty_xr.expand_dims({"time": [datetime.datetime.strptime(target_time, "%Y%m%d%H")]})
    DLAMPty_xr.to_netcdf(f'{medium_save_path}/DLAMPty_{target_time}_combined.nc')
    return

def transfer_ERA5_DLAMPty(ERA5_path, DLAMPty_path, medium_save_path, file_time, step=2, change='FCNV2'):
    file_datetime = datetime.datetime.strptime(file_time, '%Y%m%d%H')
    target_time = (file_datetime + datetime.timedelta(hours=step*3)).strftime('%Y%m%d%H')

    info = {
            "model_title": "v57 5d (SG, ra)",
            "upper_vars": ['u', 'v', 't', 'q', 'z', 'w'],
            "upper_units": ["m/s", "m/s", "K", "kg kg**-1", "m**2 s**-2", "Pa s**-1"],
            "surface_vars": ['u10', 'v10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf', 'sst_filled', 'f', 'solar', 'hgt', 'landmask', 'diurnal_sin', 'diurnal_cos', 'doy_sin', 'doy_cos'],
            "surface_units": ["m/s", "m/s", "K", "K", "Pa", "Pa", "kg m**-2", "m", 'W m**-2', 'K', '1/s', 'W m**-2', 'm', '1', '1', '1', '1', '1'],
            "pressure_levels" : [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            'coastline_color': 'darkslategray',
        }

    DLAMPty_upper = np.load(Path(DLAMPty_path, f"output_upper_{step:0>3}.npy"))
    DLAMPty_sfc = np.load(Path(DLAMPty_path, f"output_sfc_{step:0>3}.npy"))
    DLAMPty_xr = to_xarray(DLAMPty_upper, DLAMPty_sfc, info['upper_vars'], info['surface_vars'], info['upper_units'], info['surface_units'], info['pressure_levels'])
    
    print(f'DLAMPty from {DLAMPty_path}output_upper_{step:0>3}.npy')
    
    #ERA5 data
    ERA5_xr = xr.open_dataset(f'{ERA5_path}')
    ERA5_xr = ERA5_xr.isel(latitude=np.arange(40,121), longitude=np.arange(40,121)).squeeze()
    ERA5_xr = ERA5_xr.assign_coords(latitude=DLAMPty_xr.latitude, longitude=DLAMPty_xr.longitude)
    print(f'ERA5 from {ERA5_path}')
    
    
    # find TC center
    TC_center_lat = np.round(DLAMPty_xr.latitude[40].item(),2).astype(np.float32)
    TC_center_lon = np.round(DLAMPty_xr.longitude[40].item(),2).astype(np.float32)
    
    #change surface DLAMPty
    DLAMPty_xr['u10'] = change_surface_bdy(DLAMPty_xr.u10,ERA5_xr.u10)
    DLAMPty_xr['v10'] = change_surface_bdy(DLAMPty_xr.v10,ERA5_xr.v10)
    DLAMPty_xr['t2m'] = change_surface_bdy(DLAMPty_xr.t2m,ERA5_xr.t2m)
    DLAMPty_xr['sp'] = change_surface_bdy(DLAMPty_xr.sp,ERA5_xr.sp)
    DLAMPty_xr['msl'] = change_surface_bdy(DLAMPty_xr.msl,ERA5_xr.msl)
    DLAMPty_xr['tcwv'] = change_surface_bdy(DLAMPty_xr.tcwv,ERA5_xr.tcwv)

    #change upper DLAMPty
    DLAMPty_xr['u'] = change_upper_bdy(DLAMPty_xr.u, ERA5_xr.u)
    DLAMPty_xr['v'] = change_upper_bdy(DLAMPty_xr.v, ERA5_xr.v)
    DLAMPty_xr['z'] = change_upper_bdy(DLAMPty_xr.z, ERA5_xr.z)
    DLAMPty_xr['t'] = change_upper_bdy(DLAMPty_xr.t, ERA5_xr.t)
    DLAMPty_xr['q'] = change_upper_bdy(DLAMPty_xr.q, ERA5_xr.q)
    
    if change == 'all':
        DLAMPty_xr['w'] = change_upper_bdy(DLAMPty_xr.w, ERA5_xr.w)
        DLAMPty_xr['d2m'] = change_surface_bdy(DLAMPty_xr.d2m, ERA5_xr.d2m)
        DLAMPty_xr['tp'] = change_surface_bdy(DLAMPty_xr.tp, ERA5_xr.tp)
        DLAMPty_xr['mtnlwrf'] = change_surface_bdy(DLAMPty_xr.mtnlwrf, ERA5_xr.mtnlwrf)
        ERA5_sst_filled = ERA5_xr['sst'].where(~xr.ufuncs.isnan(ERA5_xr['sst']), ERA5_xr['t2m'])
        DLAMPty_xr['sst_filled'] = change_surface_bdy(DLAMPty_xr.sst_filled, ERA5_sst_filled)
        

    DLAMPty_xr = DLAMPty_xr.assign_coords(longitude = np.arange(-40,41)*0.25+TC_center_lon)
    DLAMPty_xr = DLAMPty_xr.assign_coords(latitude = np.flip(np.arange(-40,41)*0.25+TC_center_lat))
    DLAMPty_xr = DLAMPty_xr.drop_vars(["f", "solar", "hgt", "landmask", "diurnal_sin", "diurnal_cos", "doy_sin", "doy_cos","ws10","vort10"])
    DLAMPty_xr = DLAMPty_xr.drop_vars(["raw_lon", "raw_lat", "streamplot_lon", "streamplot_lat", "ws", "vort", "dewpoint", "theta_e"])
    DLAMPty_xr = DLAMPty_xr.expand_dims({"time": [datetime.datetime.strptime(target_time, "%Y%m%d%H")]})
    
    DLAMPty_xr.to_netcdf(f'{medium_save_path}/DLAMPty_{target_time}_combined.nc')
    return