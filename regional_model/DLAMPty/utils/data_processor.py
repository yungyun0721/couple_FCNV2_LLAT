import os
import numpy as np
from typing import Any
import multiprocessing as mp
from os.path import dirname, join as pathjoin
from datetime import datetime, timezone, timedelta

import xarray as xr
import xarray_regrid
import metpy.calc as mcalc
from metpy.units import units as munits
from pysolar import solar, radiation


core_count = mp.cpu_count() // 2

land_data = xr.open_dataset(pathjoin(dirname(__file__), "land.nc"))


def dict_deep_merge(dict1: dict, dict2: dict):
    """
    Recursively merge two dictionaries.
    Keys from dict2 will override or be merged into dict1.
    """
    for key, value in dict2.items():
        if (key in dict1 and isinstance(dict1[key], dict)
                and isinstance(value, dict)):
            dict_deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def load_key(opt: dict, key: str) -> Any:
    if key in opt:
        return opt[key]
    else:
        return None


def lonlat_uniformizer(
    raw_lon,
    raw_lat,
    uniformize_lonlat: bool = True,
    specify_resolution: bool | float | tuple[float, float] = False,
):
    """
    Generate longitude and latitude arrays for xarray DataArray.

    Parameters
    ----------
    raw_lon : np.ndarray
        A 2D array of longitude values.
    raw_lat : np.ndarray
        A 2D array of latitude values.
    uniformize_lonlat : bool, optional
        Whether to create uniform longitude and latitude arrays. Default is True.
    specify_resolution : bool | float | tuple[float, float], optional
        Option to specify the resolution of the longitude and latitude arrays.
        If a tuple, the first element is the resolution for longitude and
        the second for latitude. If a float, it applies to both.
        If False, resolution is not specified. Default is False.

    Returns
    -------
    lon : np.ndarray
        A 1D array of longi tude values.
    lat : np.ndarray
        A 1D array of latitude values.
    """
    # original lon and lat axis values
    lon = raw_lon.mean(0)
    lat = raw_lat.mean(1)

    if uniformize_lonlat:
        # calc the average of resolution of raw_lat
        # these step also make sure the direction of lat and lon is the same with the direction of raw_lat and raw_lon
        lat_diff = np.diff(raw_lat, axis=0).flatten()
        lat_res = lat_diff.mean()

        # calc the average of resolution of raw_lon
        lon_diff = np.diff(raw_lon, axis=1).flatten()
        lon_res = lon_diff.mean()

        if isinstance(specify_resolution, tuple):
            if specify_resolution[0] == 0:
                lon_res = 0
            elif specify_resolution[0] * lon_res > 0:
                lon_res = specify_resolution[0]
            else:
                lon_res = specify_resolution[0] * -1
            if specify_resolution[1] == 0:
                lat_res = 0
            elif specify_resolution[1] * lat_res > 0:
                lat_res = specify_resolution[1]
            else:
                lat_res = specify_resolution[1] * -1
        elif specify_resolution:
            if specify_resolution * lon_res >= 0:
                lon_res = specify_resolution
            else:
                lon_res = specify_resolution * -1
            if specify_resolution * lat_res >= 0:
                lat_res = specify_resolution
            else:
                lat_res = specify_resolution * -1
        else:
            vaild_lat_diff = lat_diff[
                (np.abs(lat_diff - np.mean(lat_diff)) < 2 * np.std(lat_diff))
            ]
            if len(vaild_lat_diff) > 0:
                lat_res = vaild_lat_diff.mean()
            vaild_lon_diff = lon_diff[
                (np.abs(lon_diff - np.mean(lon_diff)) < 2 * np.std(lon_diff))
            ]
            if len(vaild_lon_diff) > 0:
                lon_res = vaild_lon_diff.mean()
        # print(f"lat_res: {lat_res}")
        # print(f"lon_res: {lon_res}")

        # if the resolution is 0, set it to 0.25
        # if lat_res == 0:
        #     lat_res = 0.25
        # if lon_res == 0:
        #     lon_res = 0.25

        # locate the center
        lat_center = lat.mean()
        lon_center = lon.mean()
        # calc the length of lat and lon
        half_lat_length = np.floor(len(raw_lat[:, 0]) / 2)
        half_lon_length = np.floor(len(raw_lon[0, :]) / 2)
        # re-calculate the lon and lat arrays with the center value
        lat = np.linspace(
            lat_center - lat_res * half_lat_length,
            lat_center + lat_res * half_lat_length,
            len(lat),
        )
        lon = np.linspace(
            lon_center - lon_res * half_lon_length,
            lon_center + lon_res * half_lon_length,
            len(lon),
        )

    return lon, lat


def datetime_extractor(ds: xr.Dataset) -> datetime:
    """
    Extract datetime information from an xarray dataset.
    """
    # # print(ds.time.values[0])  # like: 2008-01-12T12:00:00.000000000
    # # convert the time string to a datetime object
    # time_str = str(ds.time.values[0]).split(".")[0]
    # time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S").replace(
    #     tzinfo=timezone.utc
    # )
    # return time_obj
    if "utc_time" in ds:
        return (
            ds.utc_time.values.astype("datetime64[s]")
            .astype(datetime)
            .replace(tzinfo=timezone.utc)
        )
    return (
        ds.time.values[0]
        .astype("datetime64[s]")
        .astype(datetime)
        .replace(tzinfo=timezone.utc)
    )


hourdelta_ndarray = np.vectorize(lambda x: timedelta(hours=int(x)))

sod_ndarray = np.vectorize(
    lambda dt_obj: dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second
)

get_doy = lambda dt_obj: dt_obj.timetuple().tm_yday


doy_ndarray = np.vectorize(get_doy)


def get_max_doy(ds: xr.Dataset) -> int:
    """
    Get the maximum day of year (DOY) from an xarray dataset.
    """
    last_day = datetime(year=datetime_extractor(ds).year, month=12, day=31)
    return get_doy(last_day)


def localtime_grid_from_utc(ds: xr.Dataset) -> np.ndarray:
    time = datetime_extractor(ds)
    # convert the datetime object to local time according to the timezone of each grid cell in ds
    longitude = None
    if "time" in ds.keys():
        longitude = xr.broadcast(ds.time, ds.latitude, ds.longitude)[-1]
    else:
        longitude = xr.broadcast(ds.latitude, ds.longitude)[-1]

    z = hourdelta_ndarray(np.floor(longitude / 15).astype(int))
    return z + time


def cyclical_encoder(
    original: np.ndarray, max_val: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a cyclical value (e.g. hour of day, day of year, etc.) into two components
    using a sine and cosine transformation.

    Parameters
    ----------
    original : np.ndarray
        The values to be encoded
    max_val : float
        The maximum possible value of the cyclical variable

    Returns
    -------
    tuple[np.ndarray,np.ndarray]
        Two arrays of the same shape as `original`, containing the sine and cosine
        components of the encoding.
    """
    sin_conponents = np.sin(2 * np.pi * original / max_val)
    cos_conponents = np.cos(2 * np.pi * original / max_val)

    return sin_conponents, cos_conponents


def get_radiation(date: datetime, lat: float, lon: float) -> float:
    """
    Calculate the amount of direct radiation at the given location and time.

    Parameters
    ----------
    date : datetime
        The time at which to calculate the radiation
    lat : float
        The latitude of the location
    lon : float
        The longitude of the location

    Returns
    -------
    radiation : float
        The amount of direct radiation at the given location and time
    """
    # get the solar altitude at the given location and time
    altitude = solar.get_altitude(lat, lon, date)
    # calculate the direct radiation at the given altitude
    radiation_direct = radiation.get_radiation_direct(date, altitude)
    # return the direct radiation
    return radiation_direct


def fillna_intp(da: xr.DataArray) -> xr.DataArray:
    if da.isnull().sum().values > 0:
        da = da.ffill(dim="latitude")
        if da.isnull().sum().values > 0:
            da = da.bfill(dim="latitude")
            if da.isnull().sum().values > 0:
                da = da.ffill(dim="longitude")
                if da.isnull().sum().values > 0:
                    da = da.bfill(dim="longitude")
                    if da.isnull().sum().values > 0:
                        da = da.ffill(dim="level")
                        if da.isnull().sum().values > 0:
                            da = da.bfill(dim="level")

    return da
    # res = res.interpolate_na(
    #     method="spline", dim="latitude", use_coordinate=False, keep_attrs=True
    # )
    # res = res.interpolate_na(
    #     method="spline", dim="longitude", use_coordinate=False, keep_attrs=True
    # )
    # res = res.interpolate_na(
    #     method="spline", dim="level", use_coordinate=False, keep_attrs=True
    # )
    # return (
    #     da.interpolate_na(
    #         method="spline", dim="latitude", use_coordinate=False, keep_attrs=True
    #     )
    #     .interpolate_na(
    #         method="spline", dim="longitude", use_coordinate=False, keep_attrs=True
    #     )
    #     .interpolate_na(
    #         method="spline", dim="level", use_coordinate=False, keep_attrs=True
    #     )
    # )


def compute_chunk(args):
    lat_block, lon_block, utc_time = args
    # Vectorize the get_radiation function over the block.
    vectorized_rad = np.vectorize(lambda lat, lon: get_radiation(utc_time, lat, lon))
    return vectorized_rad(lat_block, lon_block)


def radiation_from_ds(ds: xr.Dataset, utc_time: datetime) -> xr.DataArray:
    # Use raw_lat and raw_lon if available; otherwise, broadcast.
    if "raw_lat" in ds.keys() and "raw_lon" in ds.keys():
        lat = ds["raw_lat"].data
        lon = ds["raw_lon"].data
    else:
        lat, lon = xr.broadcast(ds.latitude, ds.longitude)
        lat = lat.data
        lon = lon.data

    chunk_size = 3  # Adjust this size based on your data dimensions
    tasks = []
    positions = []
    nrows, ncols = lat.shape
    for i in range(0, nrows, chunk_size):
        for j in range(0, ncols, chunk_size):
            lat_block = lat[i : i + chunk_size, j : j + chunk_size]
            lon_block = lon[i : i + chunk_size, j : j + chunk_size]
            tasks.append((lat_block, lon_block, utc_time))
            positions.append((i, j))

    # Use half of cores for processing
    with mp.Pool(processes=core_count) as pool:
        results = pool.map(compute_chunk, tasks)

    # Reassemble the computed blocks into the full result array.
    result_array = np.empty(lat.shape)
    for (i, j), block in zip(positions, results):
        r, c = block.shape
        result_array[i : i + r, j : j + c] = block

    if 'time' in ds["u10"].dims:
        result_array = np.expand_dims(result_array, ds["u10"].dims.index('time'))

    # Return as an xarray DataArray with the same coordinates/dimensions as u10.
    return fillna_intp(
        xr.DataArray(result_array, coords=ds["u10"].coords, dims=ds["u10"].dims)
    )


def recalc_additional_np(
    upper,
    sfc,
    utc_time: datetime,
    upper_vars: list[str]=['u', 'v', 't', 'q', 'z', 'w', 'ws', 'vort'],
    surface_vars: list[str]=['u10', 'v10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf', 'sst_filled', 'f', 'solar', 'hgt', 'landmask', 'diurnal_sin', 'diurnal_cos', 'doy_sin', 'doy_cos'],
    upper_units: list[str]=["m/s", "m/s", "K", "kg kg**-1", "m**2 s**-2", "Pa s**-1", "m/s", '1/s'],
    surface_units: list[str]=["m/s", "m/s", "K", "K", "Pa", "Pa", "kg m**-2", "m", 'W m**-2', 'K', '1/s', 'W m**-2', 'm', '1', '1', '1', '1', '1'],
    pressure_levels: list[int]=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    uniformize_lonlat: bool = True,
    specify_resolution: bool | float | tuple[float, float] = 0.25,
) -> dict[str, np.ndarray]:
    ds = to_xarray(
        upper,
        sfc,
        upper_vars,
        surface_vars,
        upper_units,
        surface_units,
        pressure_levels,
        utc_time=utc_time,
        uniformize_lonlat=uniformize_lonlat,
        specify_resolution=specify_resolution,
        force_additions=True,
    )
    return {
        "landmask": ds.landmask.to_numpy(),
        "hgt": ds.hgt.to_numpy(),
        "f": ds.f.to_numpy(),
        "ws10": ds.ws10.to_numpy(),
        "vort10": ds.vort10.to_numpy(),
        "diurnal_sin": ds.diurnal_sin.to_numpy(),
        "diurnal_cos": ds.diurnal_cos.to_numpy(),
        "doy_sin": ds.doy_sin.to_numpy(),
        "doy_cos": ds.doy_cos.to_numpy(),
        "solar": ds.solar.to_numpy(),
        "ws": ds.ws.to_numpy(),
        "vort": ds.vort.to_numpy(),
        "theta_e": ds.theta_e.to_numpy(),
        "dewpoint": ds.dewpoint.to_numpy(),
    }


def calc_additional_vars(ds: xr.Dataset, force: bool = False) -> xr.Dataset:
    if not "ws" in ds.keys():
        # if force or not "ws" in ds.keys():
        ds["ws"] = mcalc.wind_speed(ds.u, ds.v)
    if not "vort" in ds.keys():
        # if force or not "vort" in ds.keys():
        ds["vort"] = mcalc.vorticity(ds.u, ds.v)
    if not "theta_e" in ds.keys():
        # if force or not "theta_e" in ds.keys():
        # inflate ds.level to ds.pressure that match the shape of ds.q
        pressure = xr.broadcast(ds.level, ds.latitude, ds.longitude)[1] * munits.hPa
        if "time" in ds.keys():
            pressure = (
                xr.broadcast(ds.time, ds.level, ds.latitude, ds.longitude)[1]
                * munits.hPa
            )
        # pressure.assign_attrs["units"] = "hPa"
        ds["dewpoint"] = mcalc.dewpoint_from_specific_humidity(pressure, ds.q)
        ds["theta_e"] = mcalc.equivalent_potential_temperature(pressure, ds.t, ds.dewpoint)  # type: ignore
        ds["dewpoint"] = fillna_intp(ds["dewpoint"])
        ds["theta_e"] = fillna_intp(ds["theta_e"])
    if not "ws10" in ds.keys():
        # if force or not "ws10" in ds.keys():
        ds["ws10"] = mcalc.wind_speed(ds.u10, ds.v10)
    if not "vort10" in ds.keys():
        # if force or not "vort10" in ds.keys():
        ds["vort10"] = mcalc.vorticity(ds.u10, ds.v10)
    if force or not "f" in ds.keys():
        if "time" in ds.keys():
            ds["f"] = mcalc.coriolis_parameter(xr.broadcast(ds.time, ds.latitude, ds.longitude)[1])  # type: ignore
        else:
            lat = xr.broadcast(ds.streamplot_lat, ds.streamplot_lon)[0]
            ds["f"] = mcalc.coriolis_parameter(lat)  # type: ignore
            # ds["f"] = mcalc.coriolis_parameter(xr.broadcast(ds.latitude, ds.longitude)[0])  # type: ignore
        # elif "raw_lat" in ds.keys():
        #     ds["f"] = mcalc.coriolis_parameter(ds.raw_lat)  # type: ignore

    # fill missing values in ds sea surface temperature with t2m, as new variable
    if "sst" in ds.keys():
        if not "sst_filled_with_t2m" in ds.keys():
            ds["sst_filled_with_t2m"] = ds.sst.fillna(ds.t2m)
        if not "sst_filled" in ds.keys():
            ds["sst_filled"] = ds.sst.fillna(297.15)

    # add land data
    if force or not "landmask" in ds.keys():
        partial_land = land_data.regrid.linear(ds).broadcast_like(ds.u10)
        ds["hgt"] = partial_land.hgt
        ds["landmask"] = partial_land.landmask

    ds_with_time = "time" in ds.keys() or "utc_time" in ds.keys()
    if ds_with_time:
        if force or not "solar" in ds.keys():
            ds["solar"] = radiation_from_ds(ds, datetime_extractor(ds))

        if force or not "diurnal_sin" in ds.keys():
            localtime_grid = localtime_grid_from_utc(ds)
            coords = [ds.latitude, ds.longitude]
            dims = ["latitude", "longitude"]
            if "time" in ds.keys():
                coords = [ds.time, ds.latitude, ds.longitude]
                dims = ["time", "latitude", "longitude"]
            # add diurnal cycle variables
            # diurnal
            diurnal_sin, diurnal_cos = cyclical_encoder(
                sod_ndarray(localtime_grid), 86400
            )
            ds["diurnal_sin"] = xr.DataArray(
                diurnal_sin,
                coords=coords,
                dims=dims,
            )
            ds["diurnal_cos"] = xr.DataArray(
                diurnal_cos,
                coords=coords,
                dims=dims,
            )
            # day of year
            doy_sin, doy_cos = cyclical_encoder(
                doy_ndarray(localtime_grid), get_max_doy(ds)
            )
            ds["doy_sin"] = xr.DataArray(
                doy_sin,
                coords=coords,
                dims=dims,
            )
            ds["doy_cos"] = xr.DataArray(
                doy_cos,
                coords=coords,
                dims=dims,
            )

    return ds


def to_xarray(
    upper,
    sfc,
    upper_vars: list[str],
    surface_vars: list[str],
    upper_units: list[str],
    surface_units: list[str],
    pressure_levels: list[int],
    flip: bool = False,
    uniformize_lonlat: bool = False,
    specify_resolution: bool | float | tuple[float, float] = False,
    utc_time: datetime | None = None,
    force_additions: bool = False,
):
    """
    Convert the raw data to xarray dataset format.

    Parameters
    ----------
    upper : array_like
        The upper air variables, with shape (n_levels, n_lat, n_lon, n_vars).
    sfc : array_like
        The surface variables, with shape (n_lat, n_lon, n_vars).
    upper_vars : list[str]
        The names of the upper air variables.
    surface_vars : list[str]
        The names of the surface variables.
    upper_units : list[str]
        The units of the upper air variables.
    surface_units : list[str]
        The units of the surface variables.
    pressure_levels : list[int]
        The pressure levels of the upper air variables.
    flip : bool, optional
        Whether to flip the latitude array. Default is False.
    recreate_lonlat : bool, optional
        Whether to recreate the longitude and latitude arrays, or use the original ones. Default is False.
    specify_resolution : bool | float | tuple[float, float], optional
        Whether to specify the resolution of the longitude and latitude arrays. If False, the resolution will be calculated from the data. If float, the resolution will be set to the specified value. If tuple, the first element is the resolution of longitude and the second element is the resolution of latitude. Default is False.

    Returns
    -------
    xr.Dataset
        The xarray dataset containing the data.
    """
    if not isinstance(upper, np.ndarray):
        upper = upper.numpy()
    if not isinstance(sfc, np.ndarray):
        sfc = sfc.numpy()

    upper = np.squeeze(upper)
    sfc = np.squeeze(sfc)

    raw_lon = sfc[:, :, -2]
    raw_lat = sfc[:, :, -1]
    if flip:
        raw_lon = np.flipud(raw_lon)
        raw_lat = np.flipud(raw_lat)

    lon, lat = lonlat_uniformizer(
        raw_lon, raw_lat, uniformize_lonlat, specify_resolution
    )

    if uniformize_lonlat:
        streamplot_lat = lat
        streamplot_lon = lon

    else:
        # re-calculate the lon and lat arrays for streamplot only, to make sure they are equally spaced
        # otherwise, streamplot will not work
        streamplot_lat = raw_lat  # .mean(axis=1)
        streamplot_lon = raw_lon  # .mean(axis=0)

        max_lon = streamplot_lon.max()
        min_lon = streamplot_lon.min()
        max_lat = streamplot_lat.max()
        min_lat = streamplot_lat.min()

        if lon[0] > lon[-1]:
            streamplot_lon = np.linspace(max_lon, min_lon, len(lon))
        else:
            streamplot_lon = np.linspace(min_lon, max_lon, len(lon))
        if lat[0] > lat[-1]:
            streamplot_lat = np.linspace(max_lat, min_lat, len(lat))
        else:
            streamplot_lat = np.linspace(min_lat, max_lat, len(lat))

    data = dict()

    if utc_time is not None:
        data["utc_time"] = (
            (),
            utc_time,
        )

    for i, var in enumerate(surface_vars):
        data[var] = (
            ("latitude", "longitude"),
            np.flipud(sfc[:, :, i]) if flip else sfc[:, :, i],
            {"units": surface_units[i]},
        )

    for i, var in enumerate(upper_vars):
        data[var] = (
            ("level", "latitude", "longitude"),
            np.flip(upper[:, :, :, i], axis=1) if flip else upper[:, :, :, i],
            {"units": upper_units[i]},
        )

    data["raw_lon"] = (("latitude", "longitude"), raw_lon, {"units": "degree_east"})
    data["raw_lat"] = (("latitude", "longitude"), raw_lat, {"units": "degree_north"})
    data["streamplot_lon"] = (("longitude"), streamplot_lon, {"units": "degree_east"})
    data["streamplot_lat"] = (("latitude"), streamplot_lat, {"units": "degree_north"})

    xr_data = xr.Dataset(
        data_vars=data, coords=dict(longitude=lon, latitude=lat, level=pressure_levels)
    )

    # return xr_data
    return calc_additional_vars(xr_data, force_additions)


def combine_ncs_to_xarray(path_upper: str, path_sfc: str):
    upper = xr.open_dataset(path_upper)
    sfc = xr.open_dataset(path_sfc)
    combined = upper.merge(sfc)
    return calc_additional_vars(combined)


def npy_to_xarray(
    upper_file: str,
    sfc_file: str,
    upper_vars: list[str] = ["u", "v", "t", "q", "z", "w"],
    surface_vars: list[str] = ["u10", "v10", "t2m", "msl", "tp", "tcwv", "d2m", "sp"],
    upper_units: list[str] = ["m/s", "m/s", "K", "kg kg**-1", "m**2 s**-2", "Pa s**-1"],
    surface_units: list[str] = ["m/s", "m/s", "K", "Pa", "m", "kg m**-2", "K", "Pa"],
    pressure_levels: list[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    flip: bool = True,
    uniformize_lonlat: bool = True,
    specify_resolution: bool | float | tuple[float, float] = False,
):
    """
    Convert numpy files to xarray dataset format.

    Parameters
    ----------
    upper_file : str
        The path to the numpy file containing the upper air variables.
    sfc_file : str
        The path to the numpy file containing the surface variables.
    upper_vars : list[str]
        The names of the upper air variables.
    surface_vars : list[str]
        The names of the surface variables.
    upper_units : list[str]
        The units of the upper air variables.
    surface_units : list[str]
        The units of the surface variables.
    pressure_levels : list[int]
        The pressure levels of the upper air variables.
    flip : bool, optional
        Whether to flip the latitude array. Default is True.
    uniformize_lonlat : bool, optional
        Whether to recreate the longitude and latitude arrays, or use the original ones. Default is True.
    specify_resolution : bool | float | tuple[float, float], optional
        Whether to specify the resolution of the longitude and latitude arrays. If False, the resolution will be calculated from the data. If float, the resolution will be set to the specified value. If tuple, the first element is the resolution of longitude and the second element is the resolution of latitude. Default is False.

    Returns
    -------
    xr.Dataset
        The xarray dataset containing the data.
    """
    upper = np.load(upper_file)
    sfc = np.load(sfc_file)
    return to_xarray(
        upper,
        sfc,
        upper_vars,
        surface_vars,
        upper_units,
        surface_units,
        pressure_levels,
        flip,
        uniformize_lonlat,
        specify_resolution,
    )


def save_combined(upper_file: str, sfc_file: str, output_path: str):
    xr_data = combine_ncs_to_xarray(upper_file, sfc_file)
    xr_data.to_netcdf(output_path)
    print("Combined file saved:", output_path)


if __name__ == "__main__":
    # xr_data = npy_to_xarray(
    #     "/nwpr/wfc/com136/project/why/weather-forecast-main/onnx/out_v52e2934s334590_boundary/202001W_2020050812/output_upper_001.npy",
    #     "/nwpr/wfc/com136/project/why/weather-forecast-main/onnx/out_v52e2934s334590_boundary/202001W_2020050812/output_sfc_001.npy",
    # )
    # xr_data = combine_nc_to_xarray(
    #     "~/data/ERA5_for_TC/1_WP/2008/200801W/200801W_2008011212_20kt_upper.nc",
    #     "~/data/ERA5_for_TC/1_WP/2008/200801W/200801W_2008011212_20kt_sfc.nc",
    # )
    # xr_data = combine_nc_to_xarray(
    #     "~/data/ERA5_for_TC/1_WP/2007/200701W/200701W_2007033100_25kt_upper.nc",
    #     "~/data/ERA5_for_TC/1_WP/2007/200701W/200701W_2007033100_25kt_sfc.nc",
    # )
    # print(xr_data)
    # lat=slice(xr_data.latitude[0],xr_data.latitude[-1])
    # lon=slice(xr_data.longitude[0],xr_data.longitude[-1])
    # print(lat)
    # print(lon)
    # print(land_data.sel(latitude=lat).sel(longitude=lon))

    input_root_dir = "/nwpr/wfc/com136/data/ERA5_for_TC/1_WP/"
    output_root_dir = "/nwpr/wfc/com136/data/ERA5_TC_plus/"

    with mp.Pool(core_count) as p:
        for year in range(2007, 2021):
            # get list of TC dirs
            print("Processing year:", year)
            tc_list = os.listdir(input_root_dir + str(year))
            for tc in tc_list:
                tc_path = os.path.join(input_root_dir, str(year), tc)
                output_dir = os.path.join(output_root_dir, str(year), tc)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # get list of upper data files in each TC dir, which ended it's name with 'upper.nc'
                upper_files = [f for f in os.listdir(tc_path) if f.endswith("upper.nc")]
                for upper_file_name in upper_files:
                    print("Processing file:", upper_file_name)
                    # check corresponding sfc data file exists, which ended it's name with 'sfc.nc'
                    # replace upper with sfc in the filename and check if it exists
                    upper_file = os.path.join(tc_path, upper_file_name)
                    sfc_file = upper_file.replace("upper", "sfc")
                    if os.path.exists(sfc_file):
                        # print("SFC file found:", sfc_file)
                        # combine two files into one and save as a new file in the corresponding TC dir in output directory
                        output_name = upper_file_name.replace("upper", "combined")
                        output_path = os.path.join(
                            output_root_dir, str(year), tc, output_name
                        )
                        p.apply_async(
                            save_combined, args=(upper_file, sfc_file, output_path)
                        )
                    else:
                        print("SFC file not found:", sfc_file)
        p.close()
        p.join()
