import numpy as np

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

def rh_to_q(rh, T_K, p):
    """
    rh: 相對濕度 [%]
    T_K: 溫度 [K]
    p: 氣壓 [hPa]
    回傳 q [kg/kg]
    """
    p = (np.array(p)[:,np.newaxis]*np.ones([1,rh.shape[1]]))[:,:,np.newaxis]*np.ones([1,rh.shape[2]])
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


def transfer_FCNV2_DLAMPty_with_radius(
    FCNV2_input, 
    DLAMPty_input_upper, 
    DLAMPty_input_surface,
    DLAMP_info,
    radius=7.5):
    
    xx, yy = np.meshgrid(np.arange(81),np.arange(81))
    dis_grid = np.sqrt(((xx-40)*0.25)**2+((yy-40)*0.25)**2)
    xx = xx.reshape([-1])
    yy = yy.reshape([-1])
    dis_grid = dis_grid.reshape([-1])
    grid_mask = np.ones(xx.shape) 
    grid_mask[dis_grid>radius] = np.nan
    mask_xx = xx[~np.isnan(grid_mask)]
    mask_yy = yy[~np.isnan(grid_mask)]

    #FCNV2 data
    FCNV2_sfc_vars = [ "u10",   "v10",  "u100", "v100",   "t2m",   "sp",  "msl", "tcwv"]
    FCNV2_upper_vars = ["u", "v", "z", "t", "rh"]
    pressure_levels= [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    # find TC center
    TC_center_lat = np.round(DLAMPty_input_surface[...,-1].mean(),2).astype(np.float32)
    TC_center_lon = np.round(DLAMPty_input_surface[...,-2].mean(),2).astype(np.float32)
    # find index
    lat_change_FCNV2_index = mask_xx + np.int_((90-TC_center_lat)/0.25)-40
    lon_change_FCNV2_index = mask_yy + np.int_((TC_center_lon)/0.25)-40

    lat_index = np.arange(np.int_((90-TC_center_lat)/0.25)-40,np.int_((90-TC_center_lat)/0.25)+41)
    lon_index = np.arange(np.int_((TC_center_lon)/0.25)-40,np.int_((TC_center_lon)/0.25)+41)
    FCNV2_data = FCNV2_input.copy()

    # change FCNV2
    # surface
    sfc_replace_variables = ['u10', 'v10', 't2m', 'msl', 'sp', 'tcwv']
    for var in sfc_replace_variables:
        FCNV2_data[FCNV2_sfc_vars.index(var), lat_change_FCNV2_index, lon_change_FCNV2_index] = DLAMPty_input_surface[mask_xx,mask_yy,DLAMP_info['surface_vars'].index(var)]
    
    # upper
    upper_replace_variables = ['u', 'v', 'z', 't']
    for var in upper_replace_variables:
        FCNV2_data[(FCNV2_upper_vars.index(var)*13+8):(FCNV2_upper_vars.index(var)*13+21) , 
                    lat_change_FCNV2_index, lon_change_FCNV2_index] = DLAMPty_input_upper[:,mask_xx,mask_yy, DLAMP_info['upper_vars'].index(var)]

    DLAMPty_RH = q_to_rh(DLAMPty_input_upper[:,:,:, 3],DLAMPty_input_upper[:,:,:, 2], pressure_levels)
    FCNV2_data[60:, lat_change_FCNV2_index, lon_change_FCNV2_index] = DLAMPty_RH[:,mask_xx,mask_yy]
    # with open(f"log.txt", "a") as f:
    #     print(f"----tcwv check--------", file=f)
    #     print(f"tcwv diff{np.min(FCNV2_data[7, lat_index, :][:,lon_index]-np.array(DLAMPty_xr.tcwv))}",file=f)
    #     print(f'FCNV2 originial tcwv check {FCNV2_data_original[7, np.int_((90-TC_center_lat)/0.25), np.int_((TC_center_lon)/0.25)]}',file=f)
    #     print(f'FCNV2 tcwv check {FCNV2_data[7, np.int_((90-TC_center_lat)/0.25), np.int_((TC_center_lon)/0.25)]}',file=f)
    #     print(f'DLAMPty tcwv check {np.array(DLAMPty_xr.tcwv)[40,40]}',file=f)
    
    #change surface DLAMPty
    for var in sfc_replace_variables:
        DLAMPty_input_surface[:,:,DLAMP_info['surface_vars'].index(var)] = change_surface_bdy(
            DLAMPty_input_surface[:,:,DLAMP_info['surface_vars'].index(var)],
            FCNV2_input[FCNV2_sfc_vars.index(var), lat_index, :][:,lon_index])
    
    for var in upper_replace_variables:
        DLAMPty_input_upper[:,:,:,DLAMP_info['upper_vars'].index(var)] = change_upper_bdy(
            DLAMPty_input_upper[:,:,:,DLAMP_info['upper_vars'].index(var)],
            FCNV2_input[(FCNV2_upper_vars.index(var)*13+8):(FCNV2_upper_vars.index(var)*13+21), lat_index, :][:,:,lon_index])
    
    FCNV2_q = rh_to_q(FCNV2_input[60:73, lat_index, :][:, :,lon_index],FCNV2_data[47:60, lat_index, :][:, :,lon_index],pressure_levels)
    DLAMPty_input_upper[:,:,:, 3] = change_upper_bdy(DLAMPty_input_upper[:,:,:, 3], FCNV2_q)

    return FCNV2_data, DLAMPty_input_upper, DLAMPty_input_surface
