import yaml
import argparse
from shutil import copyfile
from os import path, makedirs
from datetime import datetime as dt, timedelta as td

import numpy as np
import onnxruntime as ort
from torch.cuda import is_available as cuda_available

from utils.datasets import ERA5TCDataset
from utils.data_processor import lonlat_uniformizer, recalc_additional_np, dict_deep_merge, load_key

"""
This script is for inferring model output with a given onnx model from a given dataset.
The model output is saved in a directory specified by the `output_path` variable.
The file name of each output is in the format of `<tc_id>_<timestamp>`.
The output is a numpy array of shape (n_level, n_lat, n_lon) for upper variables and
(n_lat, n_lon) for surface variables.

The model is assumed to have the following inputs:

- input_data: a numpy array of shape (n_level, n_lat, n_lon) for upper variables and
  (n_lat, n_lon) for surface variables.
- stat: a tuple of four numpy arrays, each of shape (n_level,) or (n_lat, n_lon).
  The first two arrays are the mean and standard deviation of the upper variables, and
  the last two arrays are the mean and standard deviation of the surface variables.
- case_name: a string in the format of `<tc_id>_<timestamp>`.

The model is assumed to have the following outputs:

- output_data: a numpy array of shape (n_level, n_lat, n_lon) for upper variables and
  (n_lat, n_lon) for surface variables.

The following options can be changed by the user:

- skip: bool, whether to skip the inference until a certain case.
- skip_till: str, the case name to skip until.
- stop_early: bool, whether to stop the inference after a certain case.
- last_case: str, the last case name to stop at.
- replace_bdry: bool, whether to replace the boundary values of the model output.
- replace_w_init_bdry: bool, whether to replace the boundary values of the model output
  for the w variable.
- dont_replace: dict[str, list[str]], a dictionary of variables and their corresponding
  dimensions to not replace. The keys of the dictionary are "upper", "surface", and "dims".
  The values of the dictionary are lists of strings, where each string is a variable name
  or a dimension name.
- fix_inference_steps: bool | int, whether to fix the number of inference steps.
  If it is an integer, the number of inference steps is fixed to that value.
  If it is False, the number of inference steps is determined by the model.
- fix_location: bool, whether to fix the location of the model output.
- uniformize_lonlat: bool, whether to uniformize the longitude and latitude of the model output.
- specify_resolution: bool | float | tuple[float, float], whether to specify the resolution of
  the model output. If it is a float, the resolution is fixed to that value.
  If it is a tuple, the resolution is fixed to the values in the tuple.
  If it is False, the resolution is determined by the model.
- plain_lat: bool | float, controls how latitude values are handled in the model output.
  If set to True, the latitude values are automatically set to their mean before entering the next step.
  If set to a float, all latitude values are set to this specific value.
  If set to False, the latitude values remain unchanged.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments to get YAML config file.
    parser.add_argument('-f','--config', default='inferencer.yaml',
                        help='Path of yaml config file')

    # Parse command line arguments.
    args = parser.parse_args()

    # Load defaults from the YAML file.
    with open(args.config) as f:
        opt = yaml.safe_load(f)

    model_config: str = load_key(opt, 'model_config')

    # Load model config from the YAML file.
    with open(model_config) as f:
        opt=dict_deep_merge(opt, yaml.safe_load(f))

    onnx_version: str = load_key(opt, 'onnx_version')
    onnx_path: str = load_key(opt, 'onnx_path')

    skip: bool = load_key(opt, 'skip')
    skip_till: str = load_key(opt, 'skip_till')
    stop_early: bool = load_key(opt, 'stop_early')
    last_case: str = load_key(opt, 'last_case')
    t0_only: bool = load_key(opt, 't0_only')
    recalc_additions: bool = load_key(opt, 'recalc_additions')
    replace_bdry: bool = load_key(opt, 'replace_bdry')
    # replace_bdry: bool = False
    replace_w_init_bdry: bool = load_key(opt, 'replace_w_init_bdry')
    dont_replace: dict[str, list[str]] | bool = load_key(opt, 'dont_replace')
    # dont_replace: bool | dict[str, list[str]] = {
    #     # "upper": ["u", "v", "w", "z"],
    #     # "surface": ["u10", "v10", "msl"],
    #     "dims": ["always means lon and lat if dims exists in keys"],
    # }
    fix_inference_steps: bool | int = load_key(opt, 'fix_inference_steps')
    fix_location: bool = load_key(opt, 'fix_location')
    uniformize_lonlat: bool = load_key(opt, 'uniformize_lonlat')
    specify_resolution: bool | float | tuple[float, float] = load_key(opt, 'specify_resolution')
    # specify_resolution: bool | float | tuple[float, float] = False
    plain_lat: bool | float = load_key(opt, 'plain_lat')
    # True: auto mode, set all lat values to it's mean before entering the next step
    # plain_lat: bool | float = True
    # float: set all lat values to this value
    # plain_lat: bool | float = 5.0
    combined_nc_input: bool = load_key(opt, 'combined_nc_input')
    dataset_path: str = load_key(opt, 'dataset_path')
    upper_vars: list[str] = load_key(opt, 'upper_vars')
    surface_vars: list[str] = load_key(opt, 'surface_vars')
    upper_units: list[str] = load_key(opt, 'upper_units')
    surface_units: list[str] = load_key(opt, 'surface_units')
    pressure_levels: list[int] = load_key(opt, 'pressure_levels')
    ingest_space_info: bool = load_key(opt, 'ingest_space_info')
    plugin_additional_vars: bool = load_key(opt, 'plugin_additional_vars')
    forecast_input: str = load_key(opt, 'forecast_input')

    start_year: int = load_key(opt, 'start_year')
    end_year: int = load_key(opt, 'end_year')

    stat_mean_file: str = load_key(opt, 'stat_mean_file')
    stat_std_file: str = load_key(opt, 'stat_std_file')

    stat_mean_upper_file: str = load_key(opt,'stat_mean_upper_file')
    stat_std_upper_file: str = load_key(opt,'stat_std_upper_file')
    stat_mean_sfc_file: str = load_key(opt,'stat_mean_sfc_file')
    stat_std_sfc_file: str = load_key(opt,'stat_std_sfc_file')

    output_path: str | None = load_key(opt,'output_path')

    if not output_path:
        output_path = f"onnx/out_{onnx_version}"

    if forecast_input:
        output_path = output_path + f"forecast_from_{path.basename(forecast_input)}"
    if replace_bdry:
        output_path = output_path + "_bdry"
    if replace_bdry and dont_replace and "dims" in dont_replace.keys(): # type: ignore
        output_path = output_path + "_SG"
    if uniformize_lonlat:
        output_path = output_path + "_uni"
    if fix_location:
        output_path = output_path + "_fixloc"
    if recalc_additions:
        output_path = output_path + "_ra"

    ####################### END OF USER INPUTS #############################

    makedirs(output_path, mode=0o755, exist_ok=True)

    # copy and replace this script to output_path
    copyfile(
        __file__,
        f"{output_path}/{parser.prog}",
    )
    copyfile(
        args.config,
        f"{output_path}/{path.basename(args.config)}",
    )
    copyfile(
        model_config,
        f"{output_path}/{path.basename(model_config)}",
    )

    # Change options here if necessary
    cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
    ort_providers = [
        ("CUDAExecutionProvider", cuda_provider_options),
        "CPUExecutionProvider",
    ]
    if not cuda_available():
        ort_providers.pop(0)
    print(f"inference with {ort_providers}")
    ort_session = ort.InferenceSession(onnx_path, providers=ort_providers)

    dataset = ERA5TCDataset(
        root_dir=dataset_path,
        start_year=start_year,
        end_year=end_year,
        upper_variables=upper_vars,
        surface_variables=surface_vars,
        data_spatial_shape=(len(pressure_levels), 81, 81),
        standardize=True,
        get_stat=True,
        combined_nc_input=combined_nc_input,
        stat_mean_file=stat_mean_file,
        stat_std_file=stat_std_file,
        stat_mean_upper_file=stat_mean_upper_file,
        stat_std_upper_file=stat_std_upper_file,
        stat_mean_sfc_file=stat_mean_sfc_file, 
        stat_std_sfc_file=stat_std_sfc_file,
        inferencing=True,
        ingest_space_info=ingest_space_info,
        plugin_additional_vars=plugin_additional_vars,
        forecast_input=forecast_input
    )

    print(f"load data")

    time_store = []
    data_store = []

    stat = []

    current_tcid = ""

    if not stop_early:
        last_case = dataset[-1][3]  # type: ignore
        # last_case = 1  # type: ignore

    print(f"Will stop at {last_case}")
    print(f"{output_path=}")

    for batch in dataset:
        input_data, output_data, stat, case_name = batch  # type: ignore
        print(case_name)
        mean_upper, std_upper, mean_surface, std_surface = (x.numpy() for x in stat)

        (tcid, timestamp) = case_name.split("_")

        if skip and case_name != skip_till and tcid != skip_till:
            print(f"Skip {case_name}...")
            continue
        elif skip:
            skip = False
            print("\n")

        stop_flag = case_name == last_case
        if stop_flag or current_tcid != tcid and len(current_tcid) > 0:
            if stop_flag:
                # need to do this in order to prevent we lost the last timestamp
                time_store.append(timestamp)
                data_store.append((input_data, output_data))
                print(f"Got {case_name}")

            # Gathered all step of one TC, start forcasting for every timestamp
            for i in range(len(time_store)):
                case_name = f"{current_tcid}_{time_store[0]}"
                sub_output_path = path.join(output_path, case_name)
                print(f"Outputing to {sub_output_path}...", flush=True)

                makedirs(sub_output_path, mode=0o755, exist_ok=True)

                forecast_steps = len(time_store)
                if fix_inference_steps:
                    forecast_steps = fix_inference_steps

                output_upper, output_surface = np.zeros(
                    (1, len(pressure_levels), 81, 81, len(upper_vars))
                ), np.zeros((1, 81, 81, len(surface_vars)))

                for ite_idx in range(forecast_steps):
                    print(f"F{ite_idx:0>2}")

                    init_data = None

                    if ite_idx == 0:
                        (input_upper, input_surface), (for_bdry_upper, for_bdry_surface) = (
                            data_store[ite_idx]
                        )
                        (output_upper, output_surface) = (
                            input_upper.numpy(),
                            input_surface.numpy(),
                        )

                        # Reverse standardization before save
                        output_upper_4save = output_upper * std_upper + mean_upper
                        output_surface_4save = output_surface * std_surface + mean_surface

                        if uniformize_lonlat:
                            lon, lat = lonlat_uniformizer(
                                output_surface_4save[:, :, -2],
                                output_surface_4save[:, :, -1],
                                uniformize_lonlat,
                                specify_resolution,
                            )

                            (
                                output_surface_4save[:, :, -2],
                                output_surface_4save[:, :, -1],
                            ) = np.meshgrid(lon, lat)

                            output_surface[:, :, -2:-1] = (
                                (output_surface_4save - mean_surface) / std_surface
                            )[:, :, -2:-1]

                        if isinstance(plain_lat, float):
                            output_surface_4save[:, :, -1] = plain_lat
                            output_surface[:, :, -1] = (
                                (output_surface_4save - mean_surface) / std_surface
                            )[:, :, -1]
                        elif plain_lat:
                            output_surface_4save[:, :, -1] = np.mean(
                                output_surface_4save[:, :, -1]
                            )
                            output_surface[:, :, -1] = (
                                (output_surface_4save - mean_surface) / std_surface
                            )[:, :, -1]

                        # model that uses Dataloader generate extra dim at axis=0, so do expand_dims to match that behavior
                        output_upper = np.expand_dims(output_upper, axis=0)
                        output_surface = np.expand_dims(output_surface, axis=0)

                        if fix_location:
                            fix_lat = output_surface[0, :, :, -1]
                            fix_lon = output_surface[0, :, :, -2]

                        if replace_w_init_bdry:
                            init_data = (output_upper.squeeze(0), output_surface.squeeze(0))

                        # Save variables
                        np.save(
                            path.join(sub_output_path, f"output_upper_{ite_idx:0>3}"),
                            output_upper_4save,
                        )
                        np.save(
                            path.join(sub_output_path, f"output_sfc_{ite_idx:0>3}"),
                            output_surface_4save,
                        )

                        continue

                    ort_inputs = {
                        # inputs are outputs from last iter
                        "input_upper": output_upper,
                        "input_surface": output_surface,
                    }

                    # Run inference
                    ort_outputs = ort_session.run(None, ort_inputs)
                    output_upper = ort_outputs[0]
                    output_surface = ort_outputs[1]

                    # Reverse standardization before save
                    output_upper_4save = (output_upper * std_upper + mean_upper).squeeze(0)
                    output_surface_4save = (
                        output_surface * std_surface + mean_surface
                    ).squeeze(0)

                    if not fix_location:
                        if uniformize_lonlat:
                            lon, lat = lonlat_uniformizer(
                                output_surface_4save[:, :, -2],
                                output_surface_4save[:, :, -1],
                                uniformize_lonlat,
                                specify_resolution,
                            )

                            (
                                output_surface_4save[:, :, -2],
                                output_surface_4save[:, :, -1],
                            ) = np.meshgrid(lon, lat)

                            output_surface[0, :, :, -2:-1] = (
                                (output_surface_4save - mean_surface) / std_surface
                            )[:, :, -2:-1]

                        if isinstance(plain_lat, float):
                            output_surface[0, :, :, -1] = (
                                (np.full(mean_surface.shape, plain_lat) - mean_surface)
                                / std_surface
                            )[:, :, -1]
                            # print(f"plain_lat set to fix value")
                        elif plain_lat:
                            output_surface[0, :, :, -1] = np.mean(
                                output_surface[0, :, :, -1]
                            )
                            # print(f"plain_lat set to mean value")

                    # Save variables
                    np.save(
                        path.join(sub_output_path, f"output_upper_{ite_idx:0>3}"),
                        output_upper_4save,
                    )
                    np.save(
                        path.join(sub_output_path, f"output_sfc_{ite_idx:0>3}"),
                        output_surface_4save,
                    )

                    if replace_w_init_bdry or replace_bdry:
                        if dont_replace:
                            # create backups
                            backup_upper = dict()
                            backup_surface = dict()
                            backup_dim = dict()

                            if "upper" in dont_replace: # type: ignore
                                for i in range(len(upper_vars)):
                                    if upper_vars[i] in dont_replace["upper"]: # type: ignore
                                        backup_upper[i] = np.copy(
                                            output_upper[0, :, :, :, i]
                                        )
                                        # print(f"{upper_vars[i]} backuped")

                            if "surface" in dont_replace: # type: ignore
                                for i in range(len(surface_vars)):
                                    if surface_vars[i] in dont_replace["surface"]: # type: ignore
                                        backup_surface[i] = np.copy(
                                            output_surface[0, :, :, i]
                                        )
                                        # print(f"{surface_vars[i]} backuped")

                            if "dims" in dont_replace: # type: ignore
                                for i in range(2):
                                    backup_dim[i - 2] = np.copy(
                                        output_surface[0, :, :, i - 2]
                                    )
                                    # print(f'{("lon", "lat")[i]} backuped')

                        if init_data:
                            for_bdry_upper, for_bdry_surface = init_data  # type: ignore
                        if not replace_w_init_bdry:
                            _, (for_bdry_upper, for_bdry_surface) = data_store[ite_idx]
                            for_bdry_upper = for_bdry_upper.numpy()
                            for_bdry_surface = for_bdry_surface.numpy()

                        # Replace boundary cells
                        output_upper[0, :, 0:8, :, :] = for_bdry_upper[:, 0:8, :, :]
                        output_upper[0, :, -8:, :, :] = for_bdry_upper[:, -8:, :, :]
                        output_upper[0, :, :, 0:8, :] = for_bdry_upper[:, :, 0:8, :]
                        output_upper[0, :, :, -8:, :] = for_bdry_upper[:, :, -8:, :]
                        output_surface[0, 0:8, :, :] = for_bdry_surface[0:8, :, :]
                        output_surface[0, -8:, :, :] = for_bdry_surface[-8:, :, :]
                        output_surface[0, :, 0:8, :] = for_bdry_surface[:, 0:8, :]
                        output_surface[0, :, -8:, :] = for_bdry_surface[:, -8:, :]

                        if dont_replace:
                            # restore backups
                            # print("Restoring...", flush=True)
                            for i, backup in backup_upper.items():
                                diff = np.mean(backup - output_upper[0, :, :, :, i])
                                # print(f"restored {upper_vars[i]} with diff: {diff}")
                                output_upper[0, :, :, :, i] = backup
                            for i, backup in backup_surface.items():
                                diff = np.mean(backup - output_surface[0, :, :, i])
                                # print(f"restored {surface_vars[i]} with diff: {diff}")
                                output_surface[0, :, :, i] = backup
                            for i, backup in backup_dim.items():
                                diff = np.mean(backup - output_surface[0, :, :, i])
                                # print(f'restored {("lon", "lat")[i]} with diff: {diff}')
                                output_surface[0, :, :, i] = backup

                    if fix_location:
                        output_surface[0, :, :, -1] = fix_lat
                        output_surface[0, :, :, -2] = fix_lon

                    if recalc_additions:
                        # reverse standardization for dealing with spacetime vars
                        output_upper_tmp = (output_upper * std_upper + mean_upper).squeeze(0)
                        output_surface_tmp = (
                            output_surface * std_surface + mean_surface
                        ).squeeze(0)

                        # timestamp looks like 2020083000
                        additionals = recalc_additional_np(
                                output_upper_tmp, output_surface_tmp, dt.strptime(time_store[0][:-3], "%Y%m%d%H")+td(hours=ite_idx*3),
                                upper_vars, surface_vars, upper_units, surface_units
                            )

                        for v in surface_vars:
                            if v in additionals:
                                i = surface_vars.index(v)
                                output_surface_tmp[:, :, i] = additionals[v]

                        for v in upper_vars:
                            if v in additionals:
                                i = upper_vars.index(v)
                                output_upper_tmp[:, :, :, i] = additionals[v]

                        # Standardize back
                        (output_surface[0, :, :, :], output_upper[0, :, :, :, :]) = (
                            (output_surface_tmp - mean_surface) / std_surface,
                            (output_upper_tmp - mean_upper) / std_upper,
                        )

                if t0_only:
                    break
                time_store.pop(0)
                data_store.pop(0)

            time_store = []
            data_store = []
            print("\n", flush=True)

        if stop_flag:
            break

        current_tcid = tcid
        time_store.append(timestamp)
        data_store.append((input_data, output_data))

        print(f"Got {case_name}", flush=True)
    print("Done!")
