from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_sync_electrolyzer_clusters import PEM_Clusters
import numpy as np
class PEM_LTA:
    def __init__(self,sys: PEM_Clusters):
        self.sys = sys
        self.plant_life_years = len(self.sys.LTA_results_annual.index.to_list())

        
    def annual_performance_for_degradation_applied_to_output(self,V_deg,V_cell,I_nom):
        #NOTE: only validated for year long simulations
        #DOES NOT WORK IF CLUSTER DIES WITHIN A YEAR
        cluster_status = self.sys.calc_cluster_status(I_nom)
        h2_multiplier = self.sys.cluster_warm_up_losses(cluster_status)

        l_sim = len(V_cell)
        Vdeg0 = 0
        refturb_schedule = np.zeros(self.plant_life_years)
        annual_power_consumption_kW = np.zeros(self.plant_life_years)
        annual_hydrogen_production_kg = np.zeros(self.plant_life_years)
        
        for y in range(int(self.plant_life_years)): #assuming sim is close to a year
            V_deg_pr_sim = Vdeg0 + V_deg
            if np.max(V_deg_pr_sim)>self.sys.d_eol:
                
                idx_dead = np.argwhere(V_deg_pr_sim>self.sys.d_eol)[0][0]
                V_deg_pr_sim = np.concatenate([V_deg_pr_sim[0:idx_dead],V_deg[idx_dead:l_sim]])
                refturb_schedule[y]=self.sys.n_stacks

            I_stack = self.sys.stack_degraded_current(I_nom,V_cell,V_deg_pr_sim)
            
            h2_kg_pr_hr_system_nom = self.sys.cell_H2_production_rate(I_stack)*self.sys.n_cells*self.sys.n_stacks
            h2_kg_pr_hr_system = h2_kg_pr_hr_system_nom*h2_multiplier
            power_consumption_kW = I_stack*(V_cell + V_deg_pr_sim)*self.sys.n_cells*self.sys.n_stacks/1000

            Vdeg0 = V_deg_pr_sim[l_sim-1]
            annual_power_consumption_kW[y] = np.sum(power_consumption_kW)
            annual_hydrogen_production_kg[y] = np.sum(h2_kg_pr_hr_system)
        annual_rated_hydrogen_production_kg = self.sys.cluster_nominal_h2_kg*l_sim #only works if sim is length of year
        annual_capacity_factor = annual_hydrogen_production_kg/annual_rated_hydrogen_production_kg

        self.sys.LTA_results_annual['Capacity Factor [-]'] = annual_capacity_factor
        self.sys.LTA_results_annual['Refurbishment Schedule [stacks replaced/year]'] = refturb_schedule
        self.sys.LTA_results_annual['Annual H2 Production [kg/year]'] = annual_hydrogen_production_kg
        self.sys.LTA_results_annual['Annual Energy Used [kWh/year]'] = annual_power_consumption_kW
        self.sys.LTA_results_annual['Annual Average Efficiency [kWh/kg]'] = annual_power_consumption_kW/annual_hydrogen_production_kg
        self.sys.LTA_results_annual['Annual Average Efficiency [%-HHV]'] = self.sys.hhv/(annual_power_consumption_kW/annual_hydrogen_production_kg)
        
        self.sys.LTA_results_average['Capacity Factor [-]'] = np.mean(annual_capacity_factor)
        self.sys.LTA_results_average['Annual H2 Production [kg/year]'] = np.mean(annual_hydrogen_production_kg)
        self.sys.LTA_results_average['Annual Energy Used [kWh/year]'] = np.mean(annual_power_consumption_kW)
        self.sys.LTA_results_average['Annual Average Efficiency [kWh/kg]'] = np.mean(annual_power_consumption_kW)/np.mean(annual_hydrogen_production_kg)
        self.sys.LTA_results_average['Annual Average Efficiency [%-HHV]'] = self.sys.hhv/(np.mean(annual_power_consumption_kW)/np.mean(annual_hydrogen_production_kg))
        
    def annual_performance_for_degradation_applied_to_input(self,V_deg,V_cell,I_nom):
        #NOTE: only validated for year long simulations
        cluster_status = self.sys.calc_cluster_status(I_nom)
        h2_multiplier = self.sys.cluster_warm_up_losses(cluster_status)
        
        l_sim = len(V_cell)
        Vdeg0 = 0
        refturb_schedule = np.zeros(self.plant_life_years)
        annual_power_consumption_kW = np.zeros(self.plant_life_years)
        annual_hydrogen_production_kg = np.zeros(self.plant_life_years)
        
        for y in range(int(self.plant_life_years)): #assuming sim is close to a year
            V_deg_pr_sim = Vdeg0 + V_deg
            if np.max(V_deg_pr_sim)>self.sys.d_eol:
                idx_dead = np.argwhere(V_deg_pr_sim>self.sys.d_eol)[0][0]
                V_deg_pr_sim = np.concatenate([V_deg_pr_sim[0:idx_dead],V_deg[idx_dead:l_sim]])
                refturb_schedule[y]=self.sys.n_stacks

            h2_kg_pr_hr_system_nom = self.sys.cell_H2_production_rate(I_nom)*self.sys.n_cells*self.sys.n_stacks
            h2_kg_pr_hr_system = h2_kg_pr_hr_system_nom*h2_multiplier
            power_consumption_kW = I_nom*(V_cell + V_deg_pr_sim)*self.sys.n_cells*self.sys.n_stacks/1000
            
            Vdeg0 = V_deg_pr_sim[l_sim-1]
            annual_power_consumption_kW[y] = np.sum(power_consumption_kW)
            annual_hydrogen_production_kg[y] = np.sum(h2_kg_pr_hr_system)
        annual_rated_hydrogen_production_kg = self.sys.cluster_nominal_h2_kg*l_sim #only works if sim is length of year
        annual_capacity_factor = annual_hydrogen_production_kg/annual_rated_hydrogen_production_kg

        self.sys.LTA_results_annual['Capacity Factor [-]'] = annual_capacity_factor
        self.sys.LTA_results_annual['Refurbishment Schedule [stacks replaced/year]'] = refturb_schedule
        self.sys.LTA_results_annual['Annual H2 Production [kg/year]'] = annual_hydrogen_production_kg
        self.sys.LTA_results_annual['Annual Energy Used [kWh/year]'] = annual_power_consumption_kW

        
    def check_dt_timestep_life(self,l_sim:int):
        n_sec_pr_hr = 3600
        n_hrs_pr_yr = 8760
        n_sec_pr_yr = n_sec_pr_hr*n_hrs_pr_yr

        sec_pr_sim = self.sys.dt*l_sim
        if sec_pr_sim==n_sec_pr_yr:
            pass
        elif sec_pr_sim<n_sec_pr_yr:
            pass
        elif sec_pr_sim>n_sec_pr_yr:
            pass
        pass