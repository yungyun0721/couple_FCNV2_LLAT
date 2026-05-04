import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import argparse

def compute_vorticity(u, v, lat, lon):
    """
    計算 850hPa 渦度 ζ = dv/dx - du/dy
    u, v shape: (lat, lon)
    lat, lon: 1D array
    """
    Re = 6371000  # 地球半徑 (m)
    lat_rad = np.radians(lat)
    dlat = np.radians(lat[1] - lat[0])
    dlon = np.radians(lon[1] - lon[0])
    
    # 經緯度網格
    lon2d, lat2d = np.meshgrid(lon, lat)

    # dx, dy in meters
    dx = Re * dlon * np.cos(lat2d * np.pi / 180)
    dy = Re * dlat

    # 中央差分法計算 dv/dx, du/dy
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)

    vort = dvdx - dudy
    return vort

# LLAT_path = '../output_data/LLAT'
# initial_time = datetime.datetime.strptime('2026041300', "%Y%m%d%H")
# save_folder = "../output_data"

def main(LLAT_path, IC_time, save_folder):
    initial_time = datetime.datetime.strptime(str(IC_time), "%Y%m%d%H")
    LLAT_data = np.load(f'{LLAT_path}/output_sfc_000h.npy')
    ws10 = np.sqrt(np.power(LLAT_data[:,:,0],2)+np.power(LLAT_data[:,:,1],2))
    position_index = np.unravel_index(np.argmax(ws10[20:-20, 20:-20]), ws10[20:-20, 20:-20].shape)
    max_ws_lat = LLAT_data[position_index[0]+20,position_index[1]+20, -1]
    max_ws_lon = LLAT_data[position_index[0]+20,position_index[1]+20, -2]
    max_ws_u10 = LLAT_data[position_index[0]+20,position_index[1]+20,  0]
    max_ws_v10 = LLAT_data[position_index[0]+20,position_index[1]+20,  1]

    # colculating vorticity
    # LLAT_upper_data = np.load(f'{LLAT_path}/forecast/output_upper_000h.npy')
    LLAT_upper_data = np.load(f'{LLAT_path}/output_upper_000h.npy')
    u850 = LLAT_upper_data[-3, 20:-20, 20:-20, 0]
    v850 = LLAT_upper_data[-3, 20:-20, 20:-20, 1]
    vort850 = compute_vorticity(u850, v850,LLAT_data[20:-20,40,-1],LLAT_data[40,20:-20,-2])

    u500 = LLAT_upper_data[ 7, 20:-20, 20:-20, 0]
    v500 = LLAT_upper_data[ 7, 20:-20, 20:-20, 1]
    vort500 = compute_vorticity(u500, v500, LLAT_data[20:-20,40,-1], LLAT_data[40,20:-20,-2])
    # 建立圓形 mask
    lon2d, lat2d = np.meshgrid(LLAT_data[40,20:-20,-2], LLAT_data[20:-20,40,-1])
    dist_deg = np.sqrt((lat2d - LLAT_data[40, 40, -1])**2 + ((lon2d - LLAT_data[40, 40, -2]) * np.cos(np.radians(LLAT_data[40, 40, -1])))**2)
    # mask = dist_deg <= 5.0
    # # 套 mask 平均
    # mean_vort = np.nanmean(vort850[mask])

    # record
    total_lat = [np.mean(LLAT_data[:, :, -1])]
    total_lon = [np.mean(LLAT_data[:, :, -2])]
    total_pressure = [np.min(LLAT_data[20:-20, 20:-20, 4])]
    total_time = [np.int_(initial_time.strftime("%Y%m%d%H"))]
    total_max_ws_lat = [max_ws_lat]
    total_max_ws_lon = [max_ws_lon]
    total_max_u10 = [max_ws_u10]
    total_max_v10 = [max_ws_v10]
    total_avg_vort850 = [np.nanmean(vort850[dist_deg <= 5.0])]
    total_avg_vort850_r3 = [np.nanmean(vort850[dist_deg <= 3.0])]
    total_avg_vort850_r2 = [np.nanmean(vort850[dist_deg <= 2.0])]
    total_avg_vort500 = [np.nanmean(vort500[dist_deg <= 5.0])]
    total_avg_vort500_r3 = [np.nanmean(vort500[dist_deg <= 3.0])]
    total_avg_vort500_r2 = [np.nanmean(vort500[dist_deg <= 2.0])]

    end_point = len(os.listdir(f'{LLAT_path}/'))/4

    # fore_i = 0
    for fore_i in range(1,int(end_point)+1):
        # LLAT_data = np.load(f'{LLAT_path}/forecast/output_sfc_{fore_i*6:0>3}h.npy')
        LLAT_data = np.load(f'{LLAT_path}/output_sfc_{fore_i*6:0>3}h.npy')
        ws10 = np.sqrt(np.power(LLAT_data[:,:,0],2)+np.power(LLAT_data[:,:,1],2))
        position_index = np.unravel_index(np.argmax(ws10[20:-20, 20:-20]), ws10[20:-20, 20:-20].shape)
        max_ws_lat = LLAT_data[position_index[0]+20,position_index[1]+20, -1]
        max_ws_lon = LLAT_data[position_index[0]+20,position_index[1]+20, -2]
        max_ws_u10 = LLAT_data[position_index[0]+20,position_index[1]+20,  0]
        max_ws_v10 = LLAT_data[position_index[0]+20,position_index[1]+20,  1]
        
        # colculating vorticity
        # LLAT_upper_data = np.load(f'{LLAT_path}/forecast/output_upper_{fore_i*6:0>3}h.npy')
        LLAT_upper_data = np.load(f'{LLAT_path}/output_upper_{fore_i*6:0>3}h.npy')
        u850 = LLAT_upper_data[-3, 20:-20, 20:-20, 0]
        v850 = LLAT_upper_data[-3, 20:-20, 20:-20, 1]
        vort850 = compute_vorticity(u850, v850,LLAT_data[20:-20,40,-1],LLAT_data[40,20:-20,-2])

        u500 = LLAT_upper_data[ 7, 20:-20, 20:-20, 0]
        v500 = LLAT_upper_data[ 7, 20:-20, 20:-20, 1]
        vort500 = compute_vorticity(u500, v500, LLAT_data[20:-20,40,-1], LLAT_data[40,20:-20,-2])
        
        # 建立圓形 mask
        lon2d, lat2d = np.meshgrid(LLAT_data[40,20:-20,-2], LLAT_data[20:-20,40,-1])
        dist_deg = np.sqrt((lat2d - LLAT_data[40, 40, -1])**2 + ((lon2d - LLAT_data[40, 40, -2]) * np.cos(np.radians(LLAT_data[40, 40, -1])))**2)
        # mask = dist_deg <= 5.0

        # # 套 mask 平均
        # mean_vort = np.nanmean(vort850[mask])

        
        target_time = initial_time+datetime.timedelta(hours=6*fore_i)
        total_lat.append(np.mean(LLAT_data[:, :, -1]))
        total_lon.append(np.mean(LLAT_data[:, :, -2]))
        total_pressure.append(np.min(LLAT_data[20:-20, 20:-20, 4]))
        total_time.append(np.int_(target_time.strftime('%Y%m%d%H')))
        total_max_ws_lat.append(max_ws_lat)
        total_max_ws_lon.append(max_ws_lon)
        total_max_u10.append(max_ws_u10)
        total_max_v10.append(max_ws_v10)
        total_avg_vort850.append(np.nanmean(vort850[dist_deg <= 5.0]))
        total_avg_vort850_r3.append(np.nanmean(vort850[dist_deg <= 3.0]))
        total_avg_vort850_r2.append(np.nanmean(vort850[dist_deg <= 2.0]))
        total_avg_vort500.append(np.nanmean(vort500[dist_deg <= 5.0]))
        total_avg_vort500_r3.append(np.nanmean(vort500[dist_deg <= 3.0]))
        total_avg_vort500_r2.append(np.nanmean(vort500[dist_deg <= 2.0]))

        
    df = pd.DataFrame({
        "time": total_time,
        "lat": np.round(total_lat,2),
        "lon": np.round(total_lon,2),
        "Pressure (hPa)": np.round(total_pressure,2),
        "max_ws_lat":np.round(total_max_ws_lat,2),
        "max_ws_lon":np.round(total_max_ws_lon,2),
        "max_ws_u10":np.round(total_max_u10,2),
        "max_ws_v10":np.round(total_max_v10,2),        
        "avg_vort850":np.round(total_avg_vort850,7),         
        "avg_vort850_r3":np.round(total_avg_vort850_r3,7),     
        "avg_vort850_r2":np.round(total_avg_vort850_r2,7),         
        "avg_vort500":np.round(total_avg_vort500,7),         
        "avg_vort500_r3":np.round(total_avg_vort500_r3,7),     
        "avg_vort500_r2":np.round(total_avg_vort500_r2,7),        
    })

    df.to_csv(f'{save_folder}/LLAT_TC_track_radius5.csv', index=False)
    print (f'finish initial time = {initial_time.strftime("%Y%m%d%H")}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LLAT_path',  required=True, help='Path to LLAT output data')
    parser.add_argument('--IC_time',  required=True, help='Initial time for IC data (ex: 2026041300)')
    parser.add_argument('--save_folder',  required=True, help='Folder to save results')
    args = parser.parse_args()  
    
    main(args.LLAT_path, str(args.IC_time), args.save_folder)     

