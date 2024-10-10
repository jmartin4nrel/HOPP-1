import pandas as pd
import numpy as np

class AlkalineCosts:
    def __init__(self,alkaline_design,plant_life_yrs:int,operation_start_year:int):
        #NOTE: Assumes that dt=3600
        self.plant_life = plant_life_yrs
        self.operation_start_year = operation_start_year
        years_of_operation = np.arange(int(self.operation_start_year),int(self.operation_start_year+self.plant_life),1)
        self.years_of_operation = [str(y) for y in years_of_operation]
        self.pf_dict = {}
        pass
    def baseline_alkaline_costs(self):
        stack_replacement_cost_percentage = 0.15
        #cheaper capex in $/kW than PEM
        uninstalled_capex = [] 
        #More maintenance compared to PEM
        variable_om = [] 
        fixed_om = [] 
        #Built on-site and larger footprint than PEM
        installation_factor = [] 
        indirect_percentage = []
        water_feedstock_cost = [] #$/gal-H2O or feedstock region
        KOH_feedstock_cost = [] #$/gram KOH

        pass
    def user_specified_costs(self):
        pass
    def add_profast_params(self,H2_Results):
        #NOTE: Assumes that dt=3600
        nameplate_kg_pr_day = 24*H2_Results["System Design"].loc["System: BOL Rated H2 Production [kg/dt]"]
        if "Performance By Year" in H2_Results.keys():
            capacity_factor = dict(zip(self.years_of_operation,H2_Results["Performance By Year"]["Capacity Factor [-]"].to_list()))
        else:
            capacity_factor = H2_Results["Simulation Summary"]["Simulation Capacity Factor [-]"]
        pf_params_dict = {}
        pf_params_dict.update({"commodity":{"initial price":100,"name":"Hydrogen","unit":"kg","escalation":0.0}})
        pf_params_dict.update({"capacity":nameplate_kg_pr_day})
        pf_params_dict.update({"long term utilization":capacity_factor})
        pf_params_dict.update({"operating life":self.plant_life})
        self.pf_dict.update({"params":pf_params_dict})

    def create_refurb_schedule(self,H2_Results):
        #TODO: finish this!
        if "Performance By Year" in H2_Results.keys():
            H2_Results["Performance By Year"]["Refurbishment Schedule [stacks replaced/year]"].to_list()

        else:
            refurb_sched = np.zeros(self.plant_life)
            years_between_replacement = int(np.floor(H2_Results["Simulation Summary"].loc["Simulation Time Until Replacement [hrs]"]/8760))
            pass

    def calc_overnight_capex(self):
        pass
    def add_profast_capital_items(self,H2_Results):
        pf_capex_dict = {}
        keys = ["name","cost","depr_type","depr_period","refurb"]
        electrolyzer_size_MW = np.round(H2_Results["System Design"].loc["System: BOL Rated Power [kW]"])/1e3
        ["Alkaline Electrolyzer System",overnight_capex,"MACRS",7,list(electrolyzer_refurbishment_schedule)]
        self.pf_dict.update({"capital_items":pf_capex_dict})
        pass
    def add_profast_fixed_costs(self,H2_Results):
        fixed_OM = [] #[$/kW-year]
        pf_fopex_dict = {}
        keys = ["name","usage","unit","cost","escalation"]
        electrolyzer_size_MW = np.round(H2_Results["System Design"].loc["System: BOL Rated Power [kW]"])/1e3

        fixed_cost_electrolysis_total = fixed_OM*electrolyzer_size_MW*1e3

        vals = ["Alkaline Electrolyzer Fixed O&M",1.0,"$/year",fixed_cost_electrolysis_total,0.0]
        
        pf_fopex_dict.update(dict(zip(keys,vals)))
        self.pf_dict.update({"fixed_costs":pf_fopex_dict})
        
    def add_profast_variable_costs(self,H2_Results):
        variable_OM = [] #[$/kWh]
        pf_vopex_dict = {}
        keys = ["name","usage","unit","cost","escalation"]
        if "Performance By Year" in H2_Results.keys():
            elec_efficiency_per_yr_kWhprkg = np.array(H2_Results["Performance By Year"]["Annual Average Efficiency [kWh/kg]"].to_list())
            annual_variable_OM_perkg = variable_OM*elec_efficiency_per_yr_kWhprkg
            total_variable_OM_perkg = dict(zip(self.years_of_operation,annual_variable_OM_perkg))
        else:
            h2_pr_sim = H2_Results["Simulation Summary"].loc["Total Hydrogen Production [kg/sim]"]
            power_pr_sim = H2_Results["Simulation Summary"].loc["Total Power Usage [kW/sim]"]
            elec_efficiency_per_yr_kWhprkg = power_pr_sim/h2_pr_sim
            total_variable_OM_perkg = variable_OM*elec_efficiency_per_yr_kWhprkg
            
        
        vals = ["Alkaline Electrolyzer Variable O&M",1.0,"$/kg",total_variable_OM_perkg,0.0]
        pf_vopex_dict.update(dict(zip(keys,vals)))
        self.pf_dict.update({"feedstocks":pf_vopex_dict})
        pass
    def add_profast_feedstock_costs(self,H2_Results):
        pf_feedstock_dict = {}
        keys = ["name","usage","unit","cost","escalation"]

        liters_pr_gallon = 3.78541 #1 gal = 3.78541 L
        gal_h2O_pr_kgH2 = H2_Results["System Design"].loc["Feedstock Usage: Liters H2O/kg-H2"]/liters_pr_gallon
        grams_KOH_pr_kgH2 = H2_Results["System Design"].loc["Feedstock Usage: Grams KOH/kg-H2"]
        water_fds_vals = ["Water",gal_h2O_pr_kgH2,"gallon-water",water_cost_pr_gal,0]
        koh_fds_vals = ["KOH",grams_KOH_pr_kgH2,"grams-KOH",KOH_cost_pr_gram,0]

        pf_feedstock_dict.update(dict(zip(keys,water_fds_vals)))
        pf_feedstock_dict.update(dict(zip(keys,koh_fds_vals)))
        self.pf_dict.update({"feedstocks":pf_feedstock_dict})

    # def create_alkaline_profast_info(self,H2_Results):
    #     if "Performance By Year" in H2_Results.keys():
    #         lta = H2_Results["Performance By Year"]
    #         lta.index = self.years_of_operation
    def create_profast_dict(self,H2_Results):
        self.add_profast_params(H2_Results)
        self.add_profast_capital_items(H2_Results)
        self.add_profast_fixed_costs(H2_Results)
        self.add_profast_variable_costs(H2_Results)
        self.add_profast_feedstock_costs(H2_Results)

    def tool_cost_scaling(self):
        pass

