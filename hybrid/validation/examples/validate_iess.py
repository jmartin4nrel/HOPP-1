import json
from pathlib import Path
import pandas as pd

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

from hybrid.validation.validate_hybrid import tune_manual, get_yaw_mismatch, tune_data

current_dir = Path(__file__).parent.absolute()

# Set API key
set_nrel_key_dot_env()

# Set wind, solar, and interconnection system info
solar_size_mw = 0.48
array_type = 0 # Fixed-angle
dc_degradation = 0

wind_size_mw = 1.5
hub_height = 80
rotor_diameter = 77
wind_dir = current_dir / ".." / "wind" / "iessGE15"
wind_power_curve = wind_dir/ "NREL_Reference_1.5MW_Turbine_Site_Level_Refactored_Hourly.csv"
wind_shear_exp = 0.15
wind_wake_model = 3 # constant wake loss, layout-independent

interconnection_size_mw = 2

technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000
                },
                'wind': {
                    'num_turbines': 1,
                    'turbine_rating_kw': wind_size_mw * 1000,
                    'hub_height': hub_height,
                    'rotor_diameter': rotor_diameter
                }}

# Get resource
flatirons_site['lat'] = 39.91
flatirons_site['lon'] = -105.22
flatirons_site['elev'] = 1835
flatirons_site['year'] = 2020
base_dir = current_dir/ ".." / ".." / ".."
resource_dir = base_dir/ "resource_files"
prices_file = resource_dir/ "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
solar_file = resource_dir/ "solar" / "39.7555_-105.2211_psmv3_60_2012.csv"
wind_file = resource_dir/ "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_100m.srw"
site = SiteInfo(flatirons_site, grid_resource_file=prices_file, solar_resource_file=solar_file, wind_resource_file=wind_file)

# Create model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.pv.value('array_type',0)
hybrid_plant.ppa_price = 0.1
hybrid_plant.pv.dc_degradation = [0]

# Enter turbine power curve
curve_data = pd.read_csv(wind_power_curve)
wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
curve_power = curve_data['Power [kW]']
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power
hybrid_plant.wind.wake_model = wind_wake_model
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.wind._system_model.Turbine.wind_resource_shear = wind_shear_exp

# Tune the model to IESS data
validation_dir = current_dir / ".."
tuning_file = validation_dir / "results" / "IESS 19 20 pv tune.csv"
hybrid_plant = tune_manual(hybrid_plant, tuning_file)

solar_dir = current_dir / ".." / "solar" / "iessFirstSolar"
tuning_files = {'pv': solar_dir / "FirstSolar_YYYY.csv",
                'wind': wind_dir / "GE1pt5MW_YYYY.csv",}
resource_files = {'pv': resource_dir / "solar" / "solar_m2_YYYY.csv",
                'wind': resource_dir / "wind" / "wind_m5_YYYY.srw",}

good_period_file = validation_dir / "hybrid" / "iessGEFS" / "GE_FirstSolar_Periods_All_Wind_Cleaned_Forward.csv"

years = [2019,2020,2021,2022]
hybrid_plant.pv.dc_degradation = [0]*len(years)

yaw_file = wind_dir / "GE Turbine Yaw Dec 2019 to 2022 gaps.csv"
tenmin_wind_file = wind_dir / "August 2012 to October 2022 M5 wind 10 min"
# hybrid_plant = get_yaw_mismatch(hybrid_plant, yaw_file, tenmin_wind_file, years)
use_dir = False

status_file = wind_dir / "GE15_IEC_validity_hourly_2019_2022"
use_status = True

hybrid_plant = tune_data(hybrid_plant, tuning_files, resource_files, good_period_file, yaw_file, status_file, years, use_status, use_dir)

hybrid_plant.simulate(1)

# Save the outputs
annual_energies = hybrid_plant.annual_energies
wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.pv
npvs = hybrid_plant.net_present_values

wind_installed_cost = hybrid_plant.wind.total_installed_cost
solar_installed_cost = hybrid_plant.pv.total_installed_cost
hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

print("Wind Installed Cost: {}".format(wind_installed_cost))
print("Solar Installed Cost: {}".format(solar_installed_cost))
print("Hybrid Installed Cost: {}".format(hybrid_installed_cost))
print("Wind NPV: {}".format(hybrid_plant.net_present_values.wind))
print("Solar NPV: {}".format(hybrid_plant.net_present_values.pv))
print("Hybrid NPV: {}".format(hybrid_plant.net_present_values.hybrid))
print("Wind + Solar Expected NPV: {}".format(wind_plus_solar_npv))


print(annual_energies)
print(npvs)
