from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_control_operation import AlkalineSupervisor
from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_system import run_alkaline_physics
from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_system import combine_results_across_clusters
import numpy as np

check_control_func = True
check_supervisor_func = True

input_signal_type = "h2"
cluster_size_mw = 1
electrolyzer_size_MW = 10
turndown_ratio = 0.25
electrolyzer_config = {
    "plant_life":30,
    "cluster_size_mw": cluster_size_mw,
    "alk_config":
        {
            "penalize_hydrogen_production": False,
            "turndown_ratio": turndown_ratio,
            }
    }
control_strategy = "even_split"
num_clusters = int(np.ceil(electrolyzer_size_MW/cluster_size_mw))
alk_config = electrolyzer_config["alk_config"]
plant_life = electrolyzer_config["plant_life"]

sup = AlkalineSupervisor(electrolyzer_size_MW,cluster_size_mw,input_signal_type,control_strategy,plant_life)
clusters = sup.create_clusters(num_clusters,cluster_size_mw,alk_config)
max_h2_pr_cluster = clusters[0].cluster_nominal_h2_kg
min_h2_pr_cluster = clusters[0].cluster_min_h2_kg

max_h2_system = max_h2_pr_cluster*num_clusters
min_h2_system = min_h2_pr_cluster*num_clusters

# MAKE STEADY HYDROGEN DEMAND INPUT
# steady_h2_demand = min_h2_system + ((max_h2_system-min_h2_system)/2)
# input_signal = steady_h2_demand*np.ones(8760)
h2_rampup = np.arange(
    0, max_h2_system, min_h2_system/4
)
h2_rampdown = np.flip(h2_rampup)
h2_dmd_in = np.concatenate((h2_rampup, h2_rampdown))
n_timesteps = len(h2_dmd_in)
t_sim = 8760
n_rep = int(np.ceil(t_sim/n_timesteps))
input_signal = np.tile(h2_dmd_in,n_rep)[:t_sim]
if check_control_func:
    res, power_consumption_total,hydrogen_production_total = sup.run(clusters,input_signal)
    H2_Results = combine_results_across_clusters(res)
    []
# RUN ALKALINE SYSTEM
if check_supervisor_func:
    H2_Results,power_consumption_total,hydrogen_production_total,res = run_alkaline_physics(input_signal,input_signal_type,electrolyzer_size_MW,electrolyzer_config,return_all_results = True)
    []