from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.validation.wind.iessGE15.ge15_wind_forecast_parse import process_wind_forecast, save_wind_forecast_SAM
from hybrid.validation.solar.iessFirstSolar.firstSolar_forecast_parse import process_solar_forecast
from hybrid.validation.examples.tune_iess import tune_iess
from hybrid.validation.validate_hybrid import tune_manual
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
from hybrid.resource import (
    SolarResource,
    WindResource
    )

# Set path to forecast files
current_dir = Path(__file__).parent.absolute()
validation_dir = current_dir/'..'/'..'
# Set path to resource files
base_dir = validation_dir/'..'/'..'
wind_res_dir = base_dir/'resource_files'/'wind'
solar_res_dir = base_dir/'resource_files'/'solar'
wind_res_fp = wind_res_dir/'wind_m5_2022.srw'
solar_res_fp = solar_res_dir/'solar_m2_2022.csv'

# Pick time period to simulate
sim_start = '07/28/22'
sim_end = '08/14/22'
sim_times = pd.date_range(sim_start, sim_end, freq='H')
save_wind_forecast_SAM({}, sim_times[0], wind_res_fp)