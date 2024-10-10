from greenheart.simulation.technologies.hydrogen.electrolysis.ALK_electrolyzer_clusters import ALK_Clusters as alk
from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_LTA import alkaline_LTA
import numpy as np
import pandas as pd
class AlkalineSupervisor:
    def __init__(self,electrolyzer_size_MW,cluster_size_MW,input_signal_type,control_strategy,plant_life):
        self.plant_life = plant_life
        num_clusters = int(np.ceil(electrolyzer_size_MW/cluster_size_MW))
        self.electrolyzer_size_MW = num_clusters*cluster_size_MW
        self.n_clusters = num_clusters
        self.input_signal_type = input_signal_type
        self.control_strategy = control_strategy

    def run(self,clusters,input_signal_profile):
        control_output = {}
        if self.input_signal_type == "power":
            print("running power control")
            if self.control_strategy == "even_split":
                power_to_clusters = self.even_split_power(input_signal_profile,clusters)
                control_power_curtailed = input_signal_profile - np.sum(power_to_clusters,axis=0)
                control_output.update({"Curtailed Power [kW]":control_power_curtailed})
        if self.input_signal_type == "h2":
            print("running h2 control")
            if self.control_strategy == "even_split":
                # H2_required_cluster_kg = self.even_split_hydrogen(hydrogen_demand_profile_kg,clusters)
                H2_required_cluster_kg = self.even_split_hydrogen(input_signal_profile,clusters)
                control_h2_demand_curtailed = input_signal_profile - np.sum(H2_required_cluster_kg,axis=0)
                control_output.update({"Curtailed H2 Demand [kg]":control_h2_demand_curtailed})
        
        results_timeseries = pd.DataFrame()
        results_summary = pd.DataFrame()
        annual_LTA = pd.DataFrame()
        average_LTA = pd.DataFrame()
        sys_design = pd.DataFrame()

        #TODO: base length off of not input_signal_profile just in case the hydrogen demand is constant
        power_consumption_total = np.zeros(len(input_signal_profile))
        hydrogen_production_total = np.zeros(len(input_signal_profile))
        for ci, cluster in enumerate(clusters):
            cluster_ID = "Cluster #{}".format(ci)
            if self.input_signal_type == "power":
                power_consumed_kW,hydrogen_produced_kg = clusters[ci].run_cluster_variable_power(power_to_clusters[ci])
                []
            if self.input_signal_type == "h2":
                power_consumed_kW,hydrogen_produced_kg = clusters[ci].run_cluster_hydrogen_demand(H2_required_cluster_kg[ci])
            
            power_consumption_total += power_consumed_kW
            hydrogen_production_total += hydrogen_produced_kg
            results_summary = pd.concat([results_summary,pd.Series(clusters[ci].simulation_results,name=cluster_ID)],axis=1)
            # pd.DataFrame(clusters[ci].timeseries_results,columns = [list(clusters[ci].timeseries_results.keys()),[cluster_ID]*len(clusters[ci].timeseries_results.keys())])
            ts_df_temp = pd.DataFrame(clusters[ci].timeseries_results)
            ts_df_temp.columns =[ts_df_temp.columns.to_list(),[cluster_ID]*len(ts_df_temp.columns.to_list())]
            results_timeseries = pd.concat([results_timeseries,ts_df_temp],axis=1)
            sys_design = pd.concat([sys_design,pd.Series(clusters[ci].BOL_design_info,name=cluster_ID)],axis=1)

            # clusters[ci].timeseries_results
            # clusters[ci].BOL_design_info
            if clusters[ci].run_LTA:
                lta_annual_df = pd.DataFrame(clusters[ci].LTA_results_annual) #DF
                lta_annual_df.columns = [clusters[ci].LTA_results_annual.columns.to_list(),[cluster_ID]*len(clusters[ci].LTA_results_annual.columns.to_list())]
                annual_LTA = pd.concat([annual_LTA,lta_annual_df],axis=1)
                clusters[ci].LTA_results_average #not updated
                average_LTA = pd.concat([average_LTA,pd.Series(clusters[ci].LTA_results_average,name=cluster_ID)],axis=1)
        if clusters[0].run_LTA:
            res = {"Controller Output":control_output,"Time Series":results_timeseries,"Summary":results_summary,"Performance By Year":annual_LTA,"Average Lifetime Performance":average_LTA,"System Design":sys_design}
        else:
            res = {"Controller Output":control_output,"Time Series":results_timeseries,"Summary":results_summary,"System Design":sys_design}
        return res, power_consumption_total,hydrogen_production_total

    def even_split_power_sequential(self):
        pass

    def even_split_hydrogen(self,hydrogen_demand_profile_kg,clusters):
        #NOTE: assumes all clusters are the same rating and capcaity
        #assumes all electrolyzer clusters are rated the same
        num_clusters_on = np.floor(hydrogen_demand_profile_kg / clusters[0].cluster_min_h2_kg)
        num_clusters_on = np.where(
            num_clusters_on > len(clusters), len(clusters), num_clusters_on
        )
        hydrogen_per_cluster = [
            hydrogen_demand_profile_kg[ti] / num_clusters_on[ti]
            if num_clusters_on[ti] > 0
            else 0
            for ti, pwr in enumerate(hydrogen_demand_profile_kg)
        ]
        hydrogen_to_active_clusters = np.array(hydrogen_per_cluster)
        hydrogen_signal_for_clusters = np.zeros((len(hydrogen_demand_profile_kg), len(clusters)))
        for i, cluster_power in enumerate(hydrogen_to_active_clusters):
            clusters_off = len(clusters) - int(num_clusters_on[i])
            no_hydrogen = np.zeros(clusters_off)
            with_hydrogen = cluster_power * np.ones(int(num_clusters_on[i]))
            tot_hydrogen = np.concatenate((with_hydrogen, no_hydrogen))
            hydrogen_signal_for_clusters[i] = tot_hydrogen


        return np.transpose(hydrogen_signal_for_clusters)
    def even_split_power(self,input_power_kW,clusters):
        #NOTE: assumes all clusters are the same rating and capcaity
        #assumes all electrolyzer clusters are rated the same
        num_clusters_on = np.floor(input_power_kW / clusters[0].min_cluster_power_kW)
        num_clusters_on = np.where(
            num_clusters_on > len(clusters), len(clusters), num_clusters_on
        )
        power_per_cluster = [
            input_power_kW[ti] / num_clusters_on[ti]
            if num_clusters_on[ti] > 0
            else 0
            for ti, pwr in enumerate(input_power_kW)
        ]
        power_to_active_clusters = np.array(power_per_cluster)
        power_signal_for_clusters = np.zeros((len(input_power_kW), len(clusters)))
        for i, cluster_power in enumerate(power_to_active_clusters):
            clusters_off = len(clusters) - int(num_clusters_on[i])
            no_power = np.zeros(clusters_off)
            with_power = cluster_power * np.ones(int(num_clusters_on[i]))
            tot_power = np.concatenate((with_power, no_power))
            power_signal_for_clusters[i] = tot_power


        return np.transpose(power_signal_for_clusters)

    def even_split_hydrogen_demand_sequential(self):
        pass

    def create_clusters(self,num_clusters:int,cluster_size_mw,alk_kwargs):
        
        clusters = []
        if isinstance(cluster_size_mw,(int,float)):
            for i in range(num_clusters):
                clusters.append(alk(cluster_size_mw,self.plant_life,**alk_kwargs))
        else:
            for i,cluster_size in enumerate(cluster_size_mw):
                clusters.append(alk(cluster_size))
        return clusters

if __name__ == "__main__":
    from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_system import combine_results_across_clusters
    electrolyzer_size_MW = 2
    cluster_size_MW = 1
    input_signal_type = "power"
    control_strategy = "even_split"
    plant_life = 30
    alk_config = {"include_degradation_penalty": True,
    "run_LTA":True,
    "debug_mode": True}
   
    num_clusters = int(np.ceil(electrolyzer_size_MW/cluster_size_MW))
    sup = AlkalineSupervisor(electrolyzer_size_MW,cluster_size_MW,input_signal_type,control_strategy,plant_life)
    clusters = sup.create_clusters(num_clusters,cluster_size_MW,alk_config)

    cluster_min_power_kw = cluster_size_MW*1e3*0.25
    power_rampup = np.arange(
        0, num_clusters * cluster_size_MW * 1e3, cluster_min_power_kw/4
    )
    power_rampdown = np.flip(power_rampup)
    power_in = np.concatenate((power_rampup, power_rampdown))
    n_timesteps = len(power_in)
    t_sim = 8760
    n_rep = int(np.ceil(t_sim/n_timesteps))
    input_signal = np.tile(power_in,n_rep)[:t_sim]
    res, power_consumption_total,hydrogen_production_total = sup.run(clusters,input_signal)
    H2_Results = combine_results_across_clusters(res)
    []
