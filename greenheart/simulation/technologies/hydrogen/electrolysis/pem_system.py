from greenheart.simulation.technologies.hydrogen.electrolysis.pem_control_operation import PEMSupervisor
import numpy as np
import pandas as pd

def combine_timeseries_across_clusters(timeseries_df):
    ts_df = pd.DataFrame()
    time_series_cols = [timeseries_df.columns.to_list()[i][0] for i in range(len(timeseries_df.columns.to_list()))]
    time_series_cols = list(np.unique(time_series_cols))
    summary_cols = ["Hydrogen Production [kg]","Power Curtailed [kW]","Power Usage [kW]","Actual Power Input [kW]","Hydrogen Losses [kg]","Cluster Status"]
    for c in summary_cols:
        temp = timeseries_df[c].sum(axis=1)
        temp.name = c
        ts_df = pd.concat([ts_df,temp],axis = 1)
    return ts_df.rename(columns = {"Cluster Status":"# Clusters Active"})

def combine_annual_LTA_across_clusters(lta_df):
    lta_res = pd.DataFrame()
    cols_average = ["Capacity Factor [-]","Annual Average Efficiency [%-HHV]","Annual Average Efficiency [kWh/kg]"]
    cols_sum = ["Annual Energy Used [kWh/year]","Refurbishment Schedule [stacks replaced/year]","Annual H2 Production [kg/year]"]
    for k in cols_sum:
        temp = lta_df[k].sum(axis=1)
        temp.name = k
        lta_res = pd.concat([lta_res,temp],axis=1)
    for k in cols_average:
        temp = lta_df[k].mean(axis=1)
        temp.name = k
        lta_res = pd.concat([lta_res,temp],axis=1)
    return lta_res

def combine_average_LTA_across_clusters(lta_df_avg):
    lta_avg = pd.DataFrame()
    
    cols_average = ["Capacity Factor [-]","Annual Average Efficiency [%-HHV]","Annual Average Efficiency [kWh/kg]","Years between stack replacement"]
    cols_sum = ["Annual Energy Used [kWh/year]","Annual H2 Production [kg/year]"]
    for k in cols_sum:
        temp = lta_df_avg.loc[k].sum()
        temp = pd.Series(temp,index=[k])
        lta_avg = pd.concat([lta_avg,temp],axis=0)
    for k in cols_average:
        temp = lta_df_avg.loc[k].mean()
        temp = pd.Series(temp,index=[k])
        lta_avg = pd.concat([lta_avg,temp],axis=0)
    # lta_avg.columns = ["Average Lifetime Performance"]
    lta_avg = pd.Series(lta_avg.to_dict()[0])
    return lta_avg

def combine_system_design(sys_des):
    design_summary = pd.DataFrame()
    sys_levels = ["System","Cluster ","Stack ","Cell "]
    
    temp = sys_des.loc["BOL Rated Cluster H2 Production [kg/dt]"].sum()
    temp = pd.Series(temp,index=["System: BOL Rated H2 Production [kg/dt]"])
    design_summary = pd.concat([design_summary,temp],axis=0)

    temp = sys_des.loc["BOL Minimum Cluster H2 Production [kg/dt]"].sum()
    temp = pd.Series(temp,index=["System: BOL Minimum H2 Production [kg/dt]"])
    design_summary = pd.concat([design_summary,temp],axis=0)

    temp = (sys_des.loc["EOL Stack Rated H2 Production [kg/dt]"]*sys_des.loc["n_stacks/cluster"]).sum()
    temp = pd.Series(temp,index=["System: EOL Rated H2 Production [kg/dt]"])
    design_summary = pd.concat([design_summary,temp],axis=0)

    temp = sys_des.loc["BOL Rated Cluster Power [kW]"].sum()
    temp = pd.Series(temp,index=["System: BOL Rated Power [kW]"])
    design_summary = pd.concat([design_summary,temp],axis=0)

    temp = len(sys_des.columns.to_list())
    temp = pd.Series(temp,index=["n_clusters/system"])
    design_summary = pd.concat([design_summary,temp],axis=0)


    sys_design_keys_init = sys_des.index.to_list()
    new_keys = []
    original_keys = []
    for level in sys_levels:
        new_keys += ["{}: ".format(level.strip()) + k.replace(level,"") for k in sys_design_keys_init if level in k]
        original_keys += [k for k in sys_design_keys_init if level in k]
    
    temp = sys_des.rename(index = dict(zip(original_keys,new_keys)))["Cluster #0"]
    design_summary = pd.concat([design_summary,temp],axis=0)
    # design_summary.columns = ["System Design"]
    design_summary = pd.Series(design_summary.to_dict()[0])
    return design_summary

def combine_results_across_clusters(res):
    H2_Results = {}
    sum_df = pd.DataFrame()
    summary_keys_sum = ["Total Curtailed Power [kW/sim]","Total Hydrogen Production [kg/sim]","Total Power Usage [kW/sim]","Total Hydrogen Losses [kg/sim]"]
    summary_keys_average = ["Simulation Time Until Replacement [hrs]","Simulation Stack Life [hrs]","On-time/sim [hrs]","Off-cycles/sim","Simulation Capacity Factor [-]"]
    for k in summary_keys_sum:
        tot = res["Summary"].loc[k].sum()
        tot = pd.Series(tot,index=[k])
        sum_df = pd.concat([sum_df,tot],axis=0)
    for k in summary_keys_average:
        avg = res["Summary"].loc[k].mean()
        avg = pd.Series(avg,index=[k])
        sum_df = pd.concat([sum_df,avg],axis=0)
    # sum_df.columns = ["Simulation Summary"]
    sum_df = pd.Series(sum_df.to_dict()[0])
    
    H2_Results.update({"Simulation Summary":sum_df})
    
    ts_df = combine_timeseries_across_clusters(res["Time Series"])
    H2_Results.update({"Time Series":ts_df})
    design_summary = combine_system_design(res["System Design"])
    H2_Results.update({"System Design":design_summary})

    if "Power" in list(res["Controller Output"].keys())[0]:
        H2_Results["Simulation Summary"]["Total Curtailed Power [kW/sim]"] += np.sum(res["Controller Output"]["Curtailed Power [kW]"])
        ts_df["Power Curtailed [kW]"] += res["Controller Output"]["Curtailed Power [kW]"]
    if "H2 Demand" in list(res["Controller Output"].keys())[0]:
        temp = pd.DataFrame(res["Controller Output"]["Curtailed H2 Demand [kg]"],columns=["Curtailed H2 Demand [kg]"])
        ts_df = pd.concat([ts_df,temp],axis=1)

    if "Performance By Year" in res.keys():
        lta_res = combine_annual_LTA_across_clusters(res["Performance By Year"])
        H2_Results.update({"Performance By Year":lta_res})
    if "Average Lifetime Performance" in res.keys():
        lta_avg = combine_average_LTA_across_clusters(res["Average Lifetime Performance"])
        H2_Results.update({"Average Lifetime Performance":lta_avg})
    
    
    return H2_Results
def run_pem_physics(input_signal,input_signal_type,electrolyzer_size_MW,electrolyzer_config,return_all_results = False):
    """1 liner desc

        longer desc:cite:`jvm-jensen1983note`

        Args:
            input_signal (np.NDarray or float): signal must correspond to input_signal_type. may either be:
                - hydrogen demand profile in kg/hr (float or array) or 
                - input power in kW
            input_signal_type (str): options are "power" or "h2"
            electrolyzer_size_MW (int): electrolyzer system capacity
            electrolyzer_config (dict):
                - cluster_size_mw (int): cluster capacity in MW
                - plant_life (int): plant life in years
                - operation_config (dict):
                    - control_strategy (str): only option is "even-split" so far
                - pem_config (dict): see docstring of optional inputs for PEM_electrolyzer_clusters
                    
            return_all_results (bool,optional): whether to return all results or just primary ones. Default is False.

        Returns:
            H2_Results (dict): dictionary of summarized results from all clusters in simulation
            power_consumption_total: electrolyzer power usage profile in kW, same length as input_signal
            hydrogen_production_total (np.NDArray): hydrogen production profile in kg, same length as input_signal
            res (dict,optional): detailed cluster-level results from simulation
        """
    pem_config = electrolyzer_config["pem_config"]
    #TODO: once more control-specific options are added, include operation_config as a used input
    # control_config = electrolyzer_config["operation_config"]
    # control_strategy = control_config["control_strategy"]
    control_strategy = "even_split"
    cluster_size_MW = electrolyzer_config["cluster_size_mw"]
    plant_life = electrolyzer_config["plant_life"]
    num_clusters = int(np.ceil(electrolyzer_size_MW/cluster_size_MW))
    sup = PEMSupervisor(electrolyzer_size_MW,cluster_size_MW,input_signal_type,control_strategy,plant_life)
    
    if input_signal_type == "h2":
        if isinstance(input_signal,(float,int)):
            input_signal = input_signal*np.ones(8760)
    if isinstance(input_signal,list):
        input_signal = np.array(input_signal)
    clusters = sup.create_clusters(num_clusters,cluster_size_MW,pem_config)
    res, power_consumption_total,hydrogen_production_total = sup.run(clusters,input_signal)
    H2_Results = combine_results_across_clusters(res)
    if return_all_results:
        return H2_Results,power_consumption_total,hydrogen_production_total,res
    else:
        return H2_Results,power_consumption_total,hydrogen_production_total
    

