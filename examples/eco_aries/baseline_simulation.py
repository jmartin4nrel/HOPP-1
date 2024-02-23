from greenheart.tools.eco.hybrid_system import run_simulation
from pytest import approx

import os

from hopp.utilities.keys import set_nrel_key_dot_env
set_nrel_key_dot_env()

import yaml
from yamlinclude import YamlIncludeConstructor 

from pathlib import Path
from ORBIT.core.library import initialize_library

dirname = os.path.dirname(__file__)
orbit_library_path = os.path.join(dirname, "input_files/")

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, 'floris/'))
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, 'turbines/'))

initialize_library(orbit_library_path)

def run_baseline():
    turbine_model = "iea_15MW"
    filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
    filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config.yaml")
    filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
    filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
    filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_wind_wave_solar_battery_baseline.yaml")

    lcoe, lcoh, _ = run_simulation(filename_hopp_config, 
                                   filename_eco_config, 
                                   filename_turbine_config, 
                                   filename_orbit_config, 
                                   filename_floris_config, 
                                   verbose=False, 
                                   show_plots=False, 
                                   save_plots=True,  
                                   use_profast=True, 
                                   incentive_option=1, 
                                   plant_design_scenario=7, 
                                   output_level=4)

if __name__ == "__main__":
    run_baseline()