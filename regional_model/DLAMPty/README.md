# DLAMP.ty

**Deep Learning for Atmospheric Modeling Predictions**

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Inference](#inference)
  - [Exporting to ONNX](#exporting-to-onnx)
  - [Plotting Results](#plotting-results)
- [Dependencies](#dependencies)
- [Scripts](#scripts)
- [Contributing](#contributing)

## Overview

DLAMP.ty is a repository designed for training and inferencing a deep learning atmospheric model called DLAMP.ty. It leverages GPU resources to perform high-performance computations, includes tools for data conversion, model training, inference, and visualization.

## Installation

To set up the project environment:

1. **Clone the Repository**:
   ```bash
   git lfs install
   git clone DLAMP.ty
   cd DLAMP.ty
   ```

2. **Set Up the Environment**:
   - Ensure you have `micromamba` installed. If not, follow the installation instructions from [Micromamba's documentation](https://mamba.readthedocs.io/en/latest/installation.html).
   - Create and activate the `ty` environment:
     ```bash
     micromamba env create -f env_building/min-win_conda_env.yaml
     micromamba activate ty
     ```

## Usage

- **Inference:**
  ```bash
  python inferencer.py -f inferencer.yaml
  ```

  modify inferencer.yaml to control inferencer behavior, examples:
  ```yaml
  # forecast from a single ERA5 ty-centered inital condiction
  # model
  model_config: onnx/v57_5d.yaml

  output_path: "path_to_save_model_output" # auto decided if not provided

  # dataset
  forecast_input: /nwpr/wfc/com136/data/sstfillc_ERA5_TC_plus/2019/201901W/201901W_2019011118_15kt_combined.nc
  combined_nc_input: True
  plugin_additional_vars: True

  # hindcast if t0_only is False
  t0_only: True

  # fix_inference_steps can be False or a positive int, need to be an int if forecasting
  fix_inference_steps: 120

  # replace_bdry need to be False if forecasting
  replace_bdry: False

  # get correct landmask etc. for every step
  recalc_additions: True

  # fix boundary conditions with initial data
  replace_w_init_bdry: False
  dont_replace: bool | dict[str, list[str]] = False

  fix_location: False
  uniformize_lonlat: True
  specify_resolution: 0.25
  # specify_resolution: bool | float | tuple[float, float] = False
  plain_lat: False
  # True: auto mode, set all lat values to it's mean before entering the next step
  # plain_lat: bool | float = True
  # float: set all lat values to this value
  # plain_lat: bool | float = 5.0

  # Change these accordingly
  skip: False
  skip_till: "202010W"
  stop_early: False
  last_case: "202026W_2020122512"

  ```

  Using CWA HPC:
  ```bash
  pjsub cwa_infer.sh
  ```
  
  - **Note:** Edit `cwa_infer.sh` if you want to use a different ONNX model or inference script.

- **Training the Model:**
  ```bash
  python train.py
  ```
  If using CWA HPC:
  ```bash
  pjsub cwa_train.sh
  ```
  This script is tailored for a specific computing cluster using the PJM job scheduler with the following parameters:
  - **Resource Unit**: `rscunit_pg01`
  - **Resource Group**: `gpu-rd-large`
  - **Number of Nodes**: 1 GPU node
  - **CPU Cores**: 32
  - **MPI Processes**: 32
  - **GPU Cards**: 8
  - **Elapse Time**: 240 hours
  - **Log Output**: Directed to `log/test.%j.out` and `log/test.%j.err`

- **Exporting to ONNX:**
  ```bash
  python export_onnx.py --ckpt_path path/to/checkpoint.ckpt --output_path path/to/output.onnx
  ```

- **Plotting Results:**
  - Use scripts in the `plotting_scripts/` directory for visualizing results.

## Scripts

Here are the scripts available in this project:

- **Training**: 
  - `train.py`
  - `cwa_train.sh` (for CWA HPC)

- **Inference**: 
  - `cwa_infer.sh` (uses `inference_onnx.py`)

- **Data Conversion**: `convert_nc_to_pt.py`

- **Exporting to ONNX**: `export_onnx.py`

- **Inference with ONNX**: 
  - `inference_onnx.py` (with or without boundary replacement)
  - `inference_onnx_161.py`

- **Plotting**: Scripts in the `plotting_scripts/` directory

**Note:** Please ignore the contents of the `outdated/` folder as they are not relevant to the current project state.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.