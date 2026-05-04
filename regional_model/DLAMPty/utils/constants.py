from dataclasses import dataclass


@dataclass(frozen=True)
class DataProperties:
    upper_vars: list[str]
    surface_vars: list[str]
    pressure_levels: list[int]


global_data_properties = DataProperties(
    upper_vars=["z", "q", "t", "u", "v"],
    surface_vars=["msl", "u10", "v10", "t2m"],
    pressure_levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
)

local_data_properties = DataProperties(
    upper_vars=["u", "v", "t", "q", "z"],
    surface_vars=["u10", "v10", "t2m", "msl"],
    pressure_levels=[50, 150, 300, 500, 700, 850, 925, 1000],
)
