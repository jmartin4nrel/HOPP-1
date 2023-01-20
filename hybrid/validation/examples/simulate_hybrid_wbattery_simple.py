import json
from pathlib import Path
import pandas as pd
import numpy as np

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

from hybrid.validation.validate_hybrid import tune_manual

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

battery_capacity_mw = 1
battery_capacity_mwh = 1

interconnection_size_mw = 1

desired_load = .3
desired_schedule = [desired_load]*8760

technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000
                },
                'wind': {
                    'num_turbines': 1,
                    'turbine_rating_kw': wind_size_mw * 1000,
                    'hub_height': hub_height,
                    'rotor_diameter': rotor_diameter
                },
                'battery': {
                    'system_capacity_kwh': battery_capacity_mw * 1000,
                    'system_capacity_kw': battery_capacity_mwh * 1000
                }}

# Get resource
flatirons_site['lat'] = 39.91
flatirons_site['lon'] = -105.22
flatirons_site['elev'] = 1835
flatirons_site['year'] = 2020
base_dir = current_dir/ ".." / ".." / ".."
resource_dir = base_dir/ "resource_files"
prices_file = resource_dir / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
solar_file = resource_dir / "solar" / "solar_m2_2019.csv"
wind_file = resource_dir / "wind" / "wind_m5_2019.srw"
site = SiteInfo(flatirons_site, 
                grid_resource_file=prices_file,
                solar_resource_file=solar_file,
                wind_resource_file=wind_file,
                desired_schedule=desired_schedule)

# Create model
solver = 'simple'
grid_charge_bool = False
dispatch_opt_dict = {'battery_dispatch':solver,'grid_charging':grid_charge_bool}
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000, dispatch_options=dispatch_opt_dict)

hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
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
hybrid_plant.wind._system_model.Turbine.wind_resource_shear = wind_shear_exp

# Tune the model to IESS data
validation_dir = current_dir / ".."
tuning_file = validation_dir / "results" / "IESS 19 20 pv tune.csv"
hybrid_plant = tune_manual(hybrid_plant, tuning_file)

hybrid_plant.simulate(1)

# Save the outputs
pv_gen = hybrid_plant.pv.generation_profile
wind_gen = hybrid_plant.wind.generation_profile
bat_gen = hybrid_plant.battery.generation_profile
bat_soc = hybrid_plant.battery.Outputs.SOC
output_df = pd.DataFrame(data=np.column_stack([pv_gen,wind_gen,bat_gen,bat_soc]),
                            columns=['PV [kW]','Wind [kW]','Battery [kW]','Battery SOC [%]'])
output_df.to_csv(validation_dir/ "results" / ("Outputs_"+solver+".csv"))

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
