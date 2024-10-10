from greenheart.simulation.technologies.hydrogen.electrolysis.pem_control_operation import PEMSupervisor
from greenheart.simulation.technologies.hydrogen.electrolysis.pem_system import run_pem_physics
from greenheart.simulation.technologies.hydrogen.electrolysis.pem_system import combine_results_across_clusters
import numpy as np
import pandas as pd

from greenheart.simulation.technologies.hydrogen.electrolysis.run_h2_PEM import run_h2_PEM
from greenheart.simulation.technologies.hydrogen.electrolysis.pem_system import run_pem_physics
import numpy as np
import pandas as pd

cluster_size_mw = 1
electrolyzer_size_MW = 10
turndown_ratio = 0.1
plant_life = 30
n_clusters = int(electrolyzer_size_MW/cluster_size_mw)
deg_penalty = True
dt = 3600
eol_eff_loss = 10
operational_hrs_until_dead = 77600
t_sim = 8760

def run_new_pem_h2_input(h2_dmd):

    input_signal_type = "h2"
    electrolyzer_config = {
    "plant_life":plant_life,
    "cluster_size_mw": electrolyzer_size_MW,
    "pem_config":
        {   
            "include_degradation_penalty":deg_penalty,
            "run_LTA":True,
            "debug_mode": True,
            "penalize_hydrogen_production": False,
            "dt":dt,
            "eol_eff_percent_loss": eol_eff_loss,
            "uptime_hours_until_eol":operational_hrs_until_dead,
            "ramp_rate_percent": 0.99,
            "turndown_ratio": turndown_ratio,
            "cold_start_delay":600,
            "anode_pressure_bar":1.01325,
            "cathode_pressure_bar":1.01325,
            }
    }

    H2_Results,power_consumption_total,hydrogen_production_total,res = run_pem_physics(h2_dmd,input_signal_type,electrolyzer_size_MW,electrolyzer_config,return_all_results = True)
    return H2_Results,power_consumption_total,hydrogen_production_total,res

def run_old_pem_h2_input(h2_dmd):
    electrical_generation_timeseries = np.zeros(8760)
    pem_params = {
        "eol_eff_percent_loss":eol_eff_loss,
        "uptime_hours_until_eol":operational_hrs_until_dead,
        "include_degradation_penalty":deg_penalty,
        "turndown_ratio":turndown_ratio,
        "dt":dt,
    }
    H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer = run_h2_PEM(electrical_generation_timeseries, 
               electrolyzer_size_MW,
               plant_life, 
               n_clusters,
               pem_control_type="basic",
               electrolyzer_direct_cost_kw=100, 
               user_defined_pem_param_dictionary=pem_params,
               grid_connection_scenario="grid-only",
               hydrogen_production_capacity_required_kgphr=h2_dmd,
               debug_mode = False,
               verbose=True)
    return H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer





max_h2_pr_cluster = 18.31*cluster_size_mw
min_h2_pr_cluster = 2.2*cluster_size_mw

max_h2_system = max_h2_pr_cluster*n_clusters
min_h2_system = min_h2_pr_cluster*n_clusters

# MAKE STEADY HYDROGEN DEMAND INPUT
steady_h2_demand = min_h2_system + ((max_h2_system-min_h2_system)/2)

new_res = run_new_pem_h2_input(steady_h2_demand)
old_res = run_old_pem_h2_input(steady_h2_demand)
[]