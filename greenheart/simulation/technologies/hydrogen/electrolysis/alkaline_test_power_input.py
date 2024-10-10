from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_control_operation import AlkalineSupervisor
from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_system import run_alkaline_physics
from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_system import combine_results_across_clusters
import numpy as np
import pandas as pd
check_control_func = True
check_supervisor_func = True

input_signal_type = "power"
cluster_size_mw = 1
electrolyzer_size_MW = 10
turndown_ratio = 0.25
electrolyzer_config = {
    "plant_life":30,
    "cluster_size_mw": cluster_size_mw,
    "alk_config":
        {
            "penalize_hydrogen_production": True,
            "turndown_ratio": turndown_ratio,
            }
    }
control_strategy = "even_split"
num_clusters = int(np.ceil(electrolyzer_size_MW/cluster_size_mw))
alk_config = electrolyzer_config["alk_config"]
plant_life = electrolyzer_config["plant_life"]

sup = AlkalineSupervisor(electrolyzer_size_MW,cluster_size_mw,input_signal_type,control_strategy,plant_life)
clusters = sup.create_clusters(num_clusters,cluster_size_mw,alk_config)

# MAKE FAKE POWER INPUT
cluster_min_power_kw = cluster_size_mw*1e3*turndown_ratio
power_rampup = np.arange(
    0, num_clusters * cluster_size_mw * 1e3, cluster_min_power_kw/4
)
power_rampdown = np.flip(power_rampup)
power_in = np.concatenate((power_rampup, power_rampdown))
n_timesteps = len(power_in)
t_sim = 8760
n_rep = int(np.ceil(t_sim/n_timesteps))
input_signal = np.tile(power_in,n_rep)[:t_sim]

if check_control_func:
    res, power_consumption_total,hydrogen_production_total = sup.run(clusters,input_signal)
    H2_Results = combine_results_across_clusters(res)
    []
# RUN ALKALINE SYSTEM
if check_supervisor_func:
    H2_Results,power_consumption_total,hydrogen_production_total,res = run_alkaline_physics(input_signal,input_signal_type,electrolyzer_size_MW,electrolyzer_config,return_all_results = True)
    []