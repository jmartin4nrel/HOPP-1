############### LOCAL FOR GEN - DO NOT COMMIT ##################
import sys
from pathlib import Path
sys.path.append('/home/gstarke/Research_Programs/HOPP/HOPP/')
################################################################

import json
from pathlib import Path
import pandas as pd

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

examples_dir = Path(__file__).parent.absolute()

# Set API key
set_nrel_key_dot_env()

# Set wind, solar, and interconnection system info
solar_size_mw = 0.48
array_type = 0 # Fixed-angle
dc_degradation = 0

wind_size_mw = 1.5
hub_height = 80
rotor_diameter = 77
wind_power_curve = examples_dir / "resource_files" / "NREL_Reference_1.5MW_Turbine_Sea_Level.csv"
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
prices_file = examples_dir.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
solar_file = examples_dir.parent / "resource_files" / "solar" / "39.7555_-105.2211_psmv3_60_2012.csv"
wind_file = examples_dir.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_100m.srw"
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
tuning_file = examples_dir / "resource_files" / "June IESS Tune.csv"
hybrid_plant.tune_manual(tuning_file)

tuning_files = {'pv': examples_dir / "resource_files" / "FirstSolar_YYYY.csv",
                'wind': examples_dir / "resource_files" / "GE1pt5MW_YYYY.csv",}
resource_files = {'pv': examples_dir / "resource_files" / "solar_m2_YYYY.csv",
                'wind': examples_dir / "resource_files" / "wind_m5_YYYY.srw",}
good_period_file = examples_dir / "resource_files" / "GE_FirstSolar_Periods_Cleaned.csv"

years = [2019,2020,2021,2022]
hybrid_plant.pv.dc_degradation = [0]*len(years)
hybrid_plant.tune_data(tuning_files, resource_files, good_period_file, years)

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
