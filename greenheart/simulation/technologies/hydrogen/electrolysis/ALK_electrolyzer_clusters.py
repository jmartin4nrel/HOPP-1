import math
import numpy as np
import sys
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import rainflow
from scipy import interpolate
from scipy.constants import R, physical_constants, convert_temperature

def stack_power_to_current(P_T,p1,p2,p3,p4,p5,p6): #calculates i-v curve coefficients given the stack power and stack temp
    pwr,tempc=P_T
    # i_stack=p1*(pwr**2) + p2*(tempc**2)+ (p3*pwr*tempc) +  (p4*pwr) + (p5*tempc) + (p6)
    i_stack=p1*(pwr**3) + p2*(pwr**2) +  (p3*pwr) + (p4*pwr**(1/2)) + p5
    return i_stack 

class ALK_Clusters:
    # num hydrogen molecules transferred per reaction
    z: int = 2 #TODO: change to z_c
    F: float = 96485.34  # Faraday's Constant (C/mol) or [As/mol]
    R: float = 8.314  # Ideal Gas Constant (J/mol/K)

    M_H: float = 1.00784  # molecular weight of Hydrogen [g/mol]
    M_H2: float = 2.016 #[g/mol]
    M_O: float = 15.999  # molecular weight of Oxygen [g/mol]
    M_O2: float = 31.999 #[g/mol]
    M_K: float = 39.0983  # molecular weight of Potassium [g/mol]
    
    lhv: float = 33.33  # lower heating value of H2 [kWh/kg]
    hhv: float = 39.41  # higher heating value of H2 [kWh/kg]
    gibbs: float = 237.24e3  # Gibbs Energy of global reaction (J/mol)
    def __init__(self,
            cluster_size_mw:int,
            plant_life:int,
            include_degradation_penalty = True,
            run_LTA = True,
            debug_mode = True,
            penalize_hydrogen_production = True,
            dt = 3600,
            eol_eff_percent_loss = 10,
            uptime_hours_until_eol = 80000,
            n_cycles_until_eol = 614,
            ramp_rate_percent = 0.2,
            turndown_ratio = 0.25,
            cold_start_delay = 1800,
        ):
        """_summary_

        Args:
            cluster_size_mw (int): cluster capacity in MW
            plant_life (int): electrolysis system plant life in years
            include_degradation_penalty (bool, optional): include degradation or not. Defaults to True.
            run_LTA (bool, optional): run life-time performance assessment. Defaults to True.
            debug_mode (bool, optional): True: return detailed results. Defaults to True.
            penalize_hydrogen_production (bool, optional): Defaults to True.
                True: degradation results in hydrogen losses. 
                False: degradation results in power increase. 
            dt (int, optional): seconds per timestep. Defaults to 3600.
            eol_eff_percent_loss (int, optional): % change from BOL rated efficiency that indicates stack needs replaced.
                Defaults to 10.
            uptime_hours_until_eol (int, optional): to customize uptime degradation rate. Defaults to 80000.
            n_cycles_until_eol (int, optional): to customize off-cycle degradation rate. Defaults to 614.
            ramp_rate_percent (float, optional): max change in current between timesteps given as a percentage. Defaults to 0.2.
            turndown_ratio (float, optional): percent of rated current to be "on". Defaults to 0.25.
            cold_start_delay (int, optional): warm-up delay in seconds. Defaults to 1800.
        """
        
        self.include_degradation_penalty = include_degradation_penalty 
        if not self.include_degradation_penalty:
            self.run_LTA = False
        else:
            self.run_LTA = run_LTA

        self.penalize_hydrogen_production = penalize_hydrogen_production
        self.dt = dt #sec/timestep
        # OPERATIONAL CONSTRIANTS
        self.ramp_rate_percent = ramp_rate_percent #percent of rated current per second
        self.turndown_ratio = turndown_ratio
        self.cold_start_delay = cold_start_delay #sec to warm up from cold
        # water_usage_mass_ratio = 10 #10 kg H2O: 1 kg H2
        
        # OPERATIONAL CONSTRIANTS
        # cell_nominal_current_density = 0.3 #[A/cm^2]
        self.nominal_current_density = 0.4 #[A/cm^2]
        
        # OPERATING CONDITIONS
        self.pressure_operating = 1 #1 # [bar] operating pressure
        T_stack = 60 #Celsius

        # CLUSTER DESIGN PARAMETERS
        #ASSUMES THAT STACK IS 1 MW THEREFORE n_stacks = cluster_size_mw
        self.n_stacks = cluster_size_mw
        
        # self.cluster_rating_kW
        
        # STACK DESIGN PARAMETERS
        self.stack_rating_kW = 1000  # 1 MW - this is reset in system_design
        self.n_cells = 882

        # CELL DESIGN PARAMETERS
        self.cell_area = 1500 #300 # [cm^2] membrane and electrode area
        self.d_em = 0.125 # [cm] electrode-membrane distance
        self.e_m = 0.05 #membrane thickness - check if used
        self.e_e = 0.2 #electrode thickness [cm]

        self.w_koh = 30 # [wt. %] can range from [25-33]
        # self.electrolyte_concentration_percent = self.w_koh / 100
        

        # CELL DEGRADATION RATES
        # self.onoff_deg_rate =  3.0726072607260716e-04 #[V/off-cycle]
        # n_cycles_until_eol = 614
        self.rate_fatigue = 1.2820512820512823e-05 #multiply by rf_track
        # self.steady_deg_rate = 5.092592592592592e-09 #V/sec
        #uptime_hours_until_eol = 5442.3

        # INITIALIZATION
        self.initalize_outputs(plant_life,run_LTA,debug_mode)
        self.m,self.M = self.create_electrolyte()
        self.system_design(T_stack)
        self.curve_coeff=self.create_power_current_curve(T_stack)

        self.eol_eff_drop = eol_eff_percent_loss/100
        self.d_eol=self.find_eol_voltage_val(eol_eff_percent_loss)
        self.BOL_design_info.update({"EOL Cell Voltage Degradation Value [V/cell]":self.d_eol})
        # CELL DEGRADATION RATES
        self.steady_deg_rate = self.reset_uptime_degradation_rate(uptime_hours_until_eol)
        self.onoff_deg_rate = self.reset_on_off_degradation_rate(n_cycles_until_eol)
        self.describe_degradation_rates()
    # def design_stack(self,T_stack,tol_kW = 1):
    #     self.system_design(T_stack)
    #     inital_error_kW = abs(self.stack_rating_kW - self.stack_rating_kW)
    #     if inital_error_kW>tol_kW:
    #         cell_power_W = self.nominal_current*self.V_cell_nominal
    def reset_on_off_degradation_rate(self,n_cycles_until_eol):
        onoff_deg_rate = self.d_eol/n_cycles_until_eol
        return onoff_deg_rate
    def reset_uptime_degradation_rate(self,uptime_hours_until_eol):
        
        steady_deg_rate = self.d_eol/(self.V_cell_nominal*uptime_hours_until_eol*3600)
        return steady_deg_rate
    def describe_degradation_rates(self):
        n_hours_until_eol_uptime = self.d_eol/(self.steady_deg_rate*self.V_cell_nominal*3600)
        n_off_cycles_until_eol = self.d_eol/(self.onoff_deg_rate)
        self.BOL_design_info.update({"Max Stack Operational Hours Until EOL":n_hours_until_eol_uptime,
        "Max Stack Off-cycles Until EOL":n_off_cycles_until_eol})
    def system_design(self,T_stack):
        self.T_stack = T_stack
        self.min_current_density = self.turndown_ratio*self.nominal_current_density
        
        self.min_current = self.min_current_density*self.cell_area
        self.nominal_current = self.nominal_current_density*self.cell_area

        # j_rated_actual = self.calc_current_density(T_stack, self.nominal_current)
        # j_min_actual = self.calc_current_density(T_stack, self.min_current)
        
        # self.rated_current_actual = j_rated_actual*self.cell_area
        # self.min_current_actual = j_min_actual*self.cell_area

        V_cell_min = self.cell_design(T_stack,self.min_current)
        V_cell_nominal = self.cell_design(T_stack,self.nominal_current)
        self.V_cell_nominal=V_cell_nominal
        self.V_cell_min = V_cell_min

        self.cell_rating_kW = self.nominal_current*V_cell_nominal/1e3
        self.stack_rating_kW = self.cell_rating_kW*self.n_cells
        self.cluster_rating_kW = self.stack_rating_kW*self.n_stacks
        
        self.min_cell_power_kW = self.min_current*V_cell_min/1e3
        self.min_stack_power_kW = self.min_cell_power_kW*self.n_cells
        self.min_cluster_power_kW = self.min_stack_power_kW*self.n_stacks

        cell_nominal_h2_kg = self.cell_H2_production_rate(T_stack,self.nominal_current)
        self.stack_nominal_h2_kg = cell_nominal_h2_kg*self.n_cells #[kg H2/dt-stack]
        self.cluster_nominal_h2_kg = self.stack_nominal_h2_kg*self.n_stacks #[kg H2/dt-cluster]

        cell_min_h2_kg = self.cell_H2_production_rate(T_stack,self.min_current)
        stack_min_h2_kg = cell_min_h2_kg*self.n_cells #[kg H2/dt-stack]
        self.cluster_min_h2_kg = stack_min_h2_kg*self.n_stacks #[kg H2/dt-cluster]

        self.current_ramp_rate = self.ramp_rate_percent*self.nominal_current*self.dt #A/dt

        #TODO: add these values to BOL design info outputs
        key_desc = ["Cluster H2 Production [kg/dt]","Cluster Power [kW]","Stack Current [A]","Cell Voltage [V/cell]","Efficiency [kWh/kg]"]
        keys = ["BOL Rated {}".format(k) for k in key_desc]
        vals = [self.cluster_nominal_h2_kg,self.cluster_rating_kW,self.nominal_current,self.V_cell_nominal,self.cluster_rating_kW/self.cluster_nominal_h2_kg]

        keys += ["BOL Minimum {}".format(k) for k in key_desc]
        vals +=[self.cluster_min_h2_kg,self.min_cluster_power_kW,self.min_current,self.V_cell_min,self.min_cluster_power_kW/self.cluster_min_h2_kg]

        self.BOL_design_info.update(dict(zip(keys,vals)))
        self.BOL_design_info.update({"n_stacks/cluster":self.n_stacks,"n_cells/stack":self.n_cells,"Minimum Stack Power [kW]":self.min_stack_power_kW})
        self.BOL_design_info.update({"Stack Operating Temperature [C]":self.T_stack})
        self.BOL_design_info.update({"Stack Anode Pressure [bar]":self.pressure_operating})
        self.BOL_design_info.update({"Stack Cathode Pressure [bar]":self.pressure_operating})
        []
# -------------------------------------------- #      
# ----- OPERATIONAL CONSTRAINTS & LOSSES ----- #
# -------------------------------------------- #    
    def initalize_outputs(self,plant_life:int,run_LTA:bool,debug_mode:bool):
        
        timeseries_keys = ["Hydrogen Production [kg]","Power Usage [kW]","Hydrogen Losses [kg]","Power Curtailed [kW]"]
        if run_LTA:
            years = np.arange(0,plant_life,1)
            years = [int(y) for y in years]
            
            lta_keys = ['Capacity Factor [-]','Refurbishment Schedule [stacks replaced/year]','Annual H2 Production [kg/year]']
            lta_keys += ['Annual Average Efficiency [kWh/kg]','Annual Average Efficiency [%-HHV]','Annual Energy Used [kWh/year]']
            x = [[None]*len(lta_keys) for _ in range(plant_life)]
            self.LTA_results_annual = pd.DataFrame(data=x,index=years,columns=lta_keys)
            self.LTA_results_average = dict(zip(lta_keys,[None]*len(lta_keys)))
            self.LTA_results_average.pop('Refurbishment Schedule [stacks replaced/year]')
            
            timeseries_keys += ["I_stack_nom","V_cell","V_deg"]
        if debug_mode:
            timeseries_keys += ["I_stack_deg","Actual Power Input [kW]","Cluster Status"]
        
        self.simulation_results = {}
        self.BOL_design_info = {}
        self.timeseries_results = dict(zip(timeseries_keys,[None]*len(timeseries_keys)))
    def find_eol_voltage_val(self,eol_eff_percent_loss):
        eol_eff_mult = (100+eol_eff_percent_loss)/100
        h2_eol_stack =self.stack_nominal_h2_kg/eol_eff_mult
        # h2_eol = s(elf.cluster_nominal_h2_kg/eol_eff_mult)/self.n_stacks
        i_eol = self.stack_reverse_faradays(h2_eol_stack)
        # V_bol = self.cell_design(self.T_stack,self.nominal_current)

        d_eol = ((self.nominal_current*self.V_cell_nominal)/i_eol) - self.V_cell_nominal
        # eol_eff_kWh_per_kg = bol_eff_kWh_per_kg*(1+eol_eff_percent_loss/100)
        
        self.BOL_design_info.update({"EOL Stack Rated H2 Production [kg/dt]":h2_eol_stack})

        return d_eol
    def x_find_eol_voltage_val(self,eol_eff_percent_loss):
        #HASN'T BEEN VERIFIED
        bol_eff_kWh_per_kg = self.cluster_rating_kW/self.cluster_nominal_h2_kg
        
        
        V_cell_nominal = self.cell_design(self.T_stack,self.nominal_current)
        eol_eff_kWh_per_kg = bol_eff_kWh_per_kg*(1+eol_eff_percent_loss/100)
        eol_power_consumed_kWh = eol_eff_kWh_per_kg*self.cluster_nominal_h2_kg
        self.BOL_design_info.update({"EOL Rated Efficiency [kWh/kg]":eol_eff_kWh_per_kg})
        v_tot_eol=eol_power_consumed_kWh*1000/(self.n_cells*self.nominal_current)
        d_eol = v_tot_eol - V_cell_nominal
        return d_eol
    def cluster_external_power_supply(self,input_external_power_kW):
        total_curtailed_power_kW = self.cluster_calc_curtailed_power(input_external_power_kW)
        power_to_electrolyzer_kW = input_external_power_kW - total_curtailed_power_kW
        return power_to_electrolyzer_kW
    def check_current_bounds(self,I_stack_nom):
        I_stack_nom_sat = np.where(I_stack_nom<self.min_current,0,I_stack_nom)
        I_stack_nom_sat = np.where(I_stack_nom_sat>self.nominal_current,self.nominal_current,I_stack_nom_sat)
        return I_stack_nom_sat
    def cluster_warm_up_losses(self,cluster_status):
        warm_up_ratio = 1 - (self.cold_start_delay/self.dt)
        #no delay at beginning of sim
        # if cluster_status[0]==0: #off at start of sim
        cluster_cycling = [0] + list(np.diff(cluster_status)) 
        # else:
        #     cluster_cycling = [0] + list(np.diff(cluster_status)) 
        cluster_cycling = np.array(cluster_cycling)
        h2_multiplier = np.where(cluster_cycling > 0, warm_up_ratio, 1)
        return h2_multiplier
    
    def add_simulation_results(self,key,value,timeseries_result:bool):
        if timeseries_result:
            if key in self.timeseries_results.keys():
                self.timeseries_results[key]= value
            # else:
            #     self.simulation_results.update({"--Total {}".format(key):sum(value)})
        else:
            self.simulation_results.update({key:value})
    
    def run_cluster(self,I_stack_nom):
        #1. check ramp rate
        I_stack_nom = self.check_ramp_rate(I_stack_nom)
        self.add_simulation_results("I_stack_nom",I_stack_nom,True)
        #2. calculate cluster on/off status (0: off, 1: on)
        cluster_status = self.calc_cluster_status(I_stack_nom)
        self.add_simulation_results("Cluster Status",cluster_status,True)
        #3. calculate hydrogen warm-up loss multipler
        h2_multiplier = self.cluster_warm_up_losses(cluster_status)
        #4. calculate un-degraded cell voltage based on nominal current
        V_cell_nom = self.cell_design(self.T_stack,I_stack_nom)
        self.add_simulation_results("V_cell",V_cell_nom,True)
        #5. calculate cell degradation degradation
        V_deg = self.cell_degradation(V_cell_nom,cluster_status)
        self.add_simulation_results("V_deg",V_deg,True)
        #6. calculate degraded current
        if self.penalize_hydrogen_production:
            I_stack = self.stack_degraded_current(I_stack_nom,V_cell_nom,V_deg)
            self.add_simulation_results("I_stack_deg",I_stack,True)
        else:
            I_stack = np.copy(I_stack_nom)
            self.add_simulation_results("I_stack_deg",I_stack,True)
        #7. calculate nominal hydrogen production
        hydrogen_produced_kg_nom = self.cell_H2_production_rate(self.T_stack,I_stack)*self.n_cells*self.n_stacks
        #8. apply hydrogen losses to hydrogen production
        hydrogen_produced_kg = hydrogen_produced_kg_nom*h2_multiplier
        self.add_simulation_results("Total Hydrogen Production [kg/sim]",np.sum(hydrogen_produced_kg),False)
        self.add_simulation_results("Hydrogen Production [kg]",hydrogen_produced_kg,True)
        #9. calculate actual power consumption
        power_consumed_kW = (I_stack*(V_cell_nom+V_deg)*self.n_cells*self.n_stacks)/1e3
        self.add_simulation_results("Total Power Usage [kW/sim]",np.sum(power_consumed_kW),False)
        self.add_simulation_results("Power Usage [kW]",power_consumed_kW,True)
        #10. calculate hydrogen losses
        hydrogen_losses_kg = hydrogen_produced_kg_nom-hydrogen_produced_kg
        self.add_simulation_results("Total Hydrogen Losses [kg/sim]",np.sum(hydrogen_losses_kg),False)
        self.add_simulation_results("Hydrogen Losses [kg]",hydrogen_losses_kg,True)
        #11. run additional post-processing
        #calculate_efficiency()
        time_between_replacement_hours = self.estimate_time_between_replacement(V_deg)
        stack_life_hours = self.estimate_stack_life(V_deg,cluster_status)
        if self.run_LTA:
            self.run_LTA_analysis(V_deg,V_cell_nom,I_stack_nom)
            avg_years_until_replacement = int(np.floor(time_between_replacement_hours/8760))
            self.LTA_results_average.update({"Years between stack replacement":avg_years_until_replacement})
        
        self.simulation_results.update({"Simulation Time Until Replacement [hrs]":time_between_replacement_hours})
        self.simulation_results.update({"Simulation Stack Life [hrs]":stack_life_hours})
        self.simulation_results.update({"Simulation Capacity Factor [-]":np.sum(hydrogen_produced_kg)/(self.cluster_nominal_h2_kg*len(hydrogen_produced_kg))})

        return power_consumed_kW,hydrogen_produced_kg

    def estimate_stack_life(self,V_deg,cluster_status):
        #based on operation (on-time)

        #[V] degradation at end of simulation
        d_sim = V_deg[-1] 
        frac_of_life_used = d_sim/self.d_eol
        operational_time_dt=np.sum(cluster_status) 
        #stack life [hrs] based on number of hours operating
        stack_life = (1/frac_of_life_used)*operational_time_dt*(self.dt/3600) #[hrs]
        # self.simulation_results.update({"Simulation Stack Life [hrs]":stack_life})
        return stack_life
    def estimate_time_between_replacement(self,V_deg):
        #based on existance (simulation time)
        d_sim = V_deg[-1] 
        frac_of_life_used = d_sim/self.d_eol
        sim_time_dt = len(V_deg) 
        #time between replacement [hrs] based on simulation length
        time_between_replacement = (1/frac_of_life_used)*sim_time_dt*(self.dt/3600)
        # self.simulation_results.update({"Simulation Time Until Replacement [hrs]":time_between_replacement})
        return time_between_replacement
    def calculate_capacity_factor(self):
        pass
    def calculate_efficiency(self,hydrogen_produced,power_consumed):
        return power_consumed/hydrogen_produced
        
    def run_LTA_analysis(self,V_deg,V_cell_nom,I_stack_nom):
        from greenheart.simulation.technologies.hydrogen.electrolysis.alkaline_LTA import alkaline_LTA
        lta = alkaline_LTA(self)
        if self.penalize_hydrogen_production:
            lta.annual_performance_for_degradation_applied_to_output(V_deg,V_cell_nom,I_stack_nom)
        else:
            lta.annual_performance_for_degradation_applied_to_input(V_deg,V_cell_nom,I_stack_nom)


    def run_cluster_variable_power(self,input_external_power_kW):
        #0a. calculate curtailed power
        total_curtailed_power_kW = self.cluster_calc_curtailed_power(input_external_power_kW)
        self.add_simulation_results("Total Curtailed Power [kW/sim]",np.sum(total_curtailed_power_kW),False)
        self.add_simulation_results("Power Curtailed [kW]",total_curtailed_power_kW,True)
        
        #0b. check cluster power input within load bounds
        input_external_power_kW = self.cluster_external_power_supply(input_external_power_kW)
        self.add_simulation_results("Actual Power Input [kW]",input_external_power_kW,True)
        #1. convert power to nominal (undegraded) current
        power_per_stack_kW = input_external_power_kW/self.n_stacks
        I_stack_nom = stack_power_to_current((power_per_stack_kW,self.T_stack),*self.curve_coeff)
        I_stack_nom = np.nan_to_num(I_stack_nom)
        I_stack_nom = self.check_current_bounds(I_stack_nom)
        
        power_consumed_kW,hydrogen_produced_kg = self.run_cluster(I_stack_nom)
        return power_consumed_kW,hydrogen_produced_kg

        
        
    def run_cluster_hydrogen_demand(self,H2_required_cluster_kg):
        I_stack_nom,input_external_power_kW = self.calc_current_power_required_for_hydrogen_demand(H2_required_cluster_kg)
        #0b. check cluster power input within load bounds
        total_curtailed_power_kW = self.cluster_calc_curtailed_power(input_external_power_kW)
        self.add_simulation_results("Total Curtailed Power [kW/sim]",np.sum(total_curtailed_power_kW),False)
        self.add_simulation_results("Power Curtailed [kW]",total_curtailed_power_kW,True)

        input_external_power_kW = self.cluster_external_power_supply(input_external_power_kW)
        self.add_simulation_results("Actual Power Input [kW]",input_external_power_kW,True)

        power_consumed_kW,hydrogen_produced_kg = self.run_cluster(I_stack_nom)
        return power_consumed_kW,hydrogen_produced_kg

        


    def calc_current_power_required_for_hydrogen_demand(self,H2_required_cluster_kg):
        H2_required_per_stack_kg = H2_required_cluster_kg/self.n_stacks
        I_reqd = self.stack_reverse_faradays(H2_required_per_stack_kg)
        #Saturate current to rated
        I_reqd = np.where(I_reqd>self.nominal_current,self.nominal_current,I_reqd)
        V_reqd = self.cell_design(self.T_stack,I_reqd)
        V_deg_est = self.estimate_cell_degradation_from_demand(H2_required_cluster_kg)
        power_reqd_kW = (I_reqd*(V_reqd+V_deg_est)*self.n_cells*self.n_stacks)/1e3
        return I_reqd,power_reqd_kW
        
    
    def check_ramp_rate(self,I_stack):
        
        if self.current_ramp_rate>self.nominal_current:
            return I_stack
        else:
            dI = np.abs(np.diff(I_stack))
            if np.max(dI)>self.current_ramp_rate:
                i_stack_n = I_stack[0]
                for i in range(1,len(I_stack)):
                    di = I_stack[i]-i_stack_n
                    if di>self.current_ramp_rate:
                        I_stack[i] = i_stack_n + di
                    i_stack_n = I_stack[i]

            return I_stack
# ------------------------------------- #      
# ----- CLUSTER - LEVEL EQUATIONS ----- #
# ------------------------------------- #     

    def calc_cluster_status(self,I_stack):
        #NOTE: should on/off be determined by current or current density or power?
        # I_stack = np.nan_to_num(I_stack)
        cluster_status=np.where(I_stack<self.min_current,0,1)

        return cluster_status
    def cluster_calc_curtailed_power(self,input_external_power_kW):
        excess_power_curtailed_kW = np.where(input_external_power_kW > self.cluster_rating_kW,\
        input_external_power_kW - self.cluster_rating_kW,0)

        #minimum_power_kW = self.turndown_ratio*self.cluster_rating_kW
        below_turndown_power_curtailed_kW = np.where(input_external_power_kW<self.min_cluster_power_kW,input_external_power_kW,0)
        total_curtailed_power_kW = below_turndown_power_curtailed_kW + excess_power_curtailed_kW
        return total_curtailed_power_kW
# ----------------------------------- #
# ----- STACK - LEVEL EQUATIONS ----- #
# ----------------------------------- #
    # def run_stack(self,T_stack,I_stack):
    #     cluster_status = self.calc_cluster_status(I_stack)
    #     stack_h2_production_rate = self.cell_H2_production_rate(T_stack,I_stack)*self.n_cells
    #     pass
    def create_power_current_curve(self,T_stack):
        #NOTE: should this be moved to higher level (like AlkalineSupervisor?)
        #TODO: add in curve error
        
        dA = 10
        dT = 5
        current_range = np.arange(self.min_current,self.nominal_current+dA,dA) 
        temp_range = np.arange(T_stack-dT,T_stack+dT,dT)
        # np.piecewise(x, [x < 0, x >= 0], [-1, 1])
        powers = np.zeros(len(current_range)*len(temp_range))
        currents = np.zeros(len(current_range)*len(temp_range))
        temps_C = np.zeros(len(current_range)*len(temp_range))
        cell_voltage = np.zeros(len(current_range)*len(temp_range))
        current_density = np.zeros(len(current_range)*len(temp_range))
        idx = 0
        for i in range(len(current_range)):
            
            for t in range(len(temp_range)):
                current_density[idx] = self.calc_current_density(temp_range[t],current_range[i])
                cell_voltage[idx] = self.cell_design(temp_range[t],current_range[i])
                powers[idx] = current_range[i]*self.cell_design(temp_range[t],current_range[i])*self.n_cells*(1e-3) #stack power
                currents[idx] = current_range[i]
                temps_C[idx] = temp_range[t]
                idx = idx+1
        df=pd.DataFrame({'Power':powers,'Current':currents,'Temp':temps_C}) #added
        temp_oi_idx = df.index[df['Temp']==T_stack]   
        curve_coeff, curve_cov = scipy.optimize.curve_fit(stack_power_to_current, (df['Power'][temp_oi_idx].values,df['Temp'][temp_oi_idx].values), df['Current'][temp_oi_idx].values, p0=(1.0,1.0,1.0,1.0,1.0,1.0))

        df_dbg=pd.DataFrame({'T [C]':temps_C,'I [A]':currents,'V_cell':cell_voltage,"J [A/cm^2]":current_density,'Power [kW/cell]':powers}) #added
        df_dbg2 = df_dbg[df_dbg["T [C]"]==T_stack]

        i_actual = df['Current'][temp_oi_idx].values
        i_estimated = stack_power_to_current((df['Power'][temp_oi_idx].values,df['Temp'][temp_oi_idx].values),*curve_coeff)
        i_error = i_actual - i_estimated
        V_actual = self.cell_design(T_stack,i_actual)
        V_estimated = self.cell_design(T_stack,i_estimated)
        P_actual = i_actual*V_actual*self.n_cells*(1e-3)
        P_estimated = i_estimated*V_estimated*self.n_cells*(1e-3)
        P_error = P_actual - P_estimated
        return curve_coeff
    def stack_degraded_current(self,I_stack,V_init,V_deg):
        """1 liner desc

        longer desc:cite:`jvm-jensen1983note`

        Args:
            I_stack (_type_): _description_
            V_init (_type_): _description_
            V_deg (_type_): _description_

        Returns:
            _type_: _description_
        
        """
        #current decrease - same power
        # I_in = calc_current((power_input_kW,self.T_C), *self.curve_coeff)
        eff_mult = (V_init + V_deg)/V_init #(1 + eff drop)
        I_deg = I_stack/eff_mult

        return I_deg
    def stack_degraded_power(self,I_nom,V_init,V_deg):
        #power increase - same current
        P_deg_kW = I_nom*(V_init+V_deg)*self.n_cells/1e3
        return P_deg_kW
# ---------------------------------- #
# ----- CELL - LEVEL EQUATIONS ----- #
# ---------------------------------- #
    def create_electrolyte(self):
        electrolyte_concentration_percent = self.w_koh / 100
        solution_weight_g = 1000
        density_of_water = 1  # [g/mL] #TODO: could be temperature dependent
        density_of_KOH = 2.12  # [g/mL] #TODO: could be temperature dependent

        self.M_KOH = self.M_O + self.M_H + self.M_K  # [g/mol]
        grams_of_solute = solution_weight_g * (electrolyte_concentration_percent)
        moles_of_solute = grams_of_solute / self.M_KOH  # [mols of solute / solution]
        # solvent is water
        self.M_H2O = 2 * self.M_H + self.M_O  # [g/mol]
        grams_of_solvent = solution_weight_g * (
            1 - electrolyte_concentration_percent
        )
        kg_of_solvent = (1 / 1000) * grams_of_solvent

        volume_of_water = grams_of_solvent / density_of_water  # mL
        volume_of_KOH = grams_of_solute / density_of_KOH  # mL
        volume_of_solution = (volume_of_water + volume_of_KOH) / 1000  # L

        # molality = mol of solute / kg of solvent
        molality = moles_of_solute / kg_of_solvent  # mol/kg
        # molarity is mol/kg = mol of solute / L of solution
        molarity = moles_of_solute / volume_of_solution  # mol/L
        # % solution = amount of solute / amount of solution

        self.m = molality  # NOTE: THIS HAS BEEN VALIDATED
        self.M = molarity  # NOTE: THIS HAS BEEN VALIDATED

        self.feedstock_usage()
        return molality,molarity
    

    def feedstock_usage(self):
        #TODO: finish this based on percent weight formula and half-cell reaction equations
        H2_production_kg = 1
        #solvent is water
        #solute is KOH
        # molality is self.m #mol KOH / kg H2O
        #cathode: 2H2O + 2e- -> H2 + 2OH-
        #anode: 4OH- -> O2 + 2*H2O + 4e-
        #https://pubs.acs.org/doi/10.1021/acs.accounts.3c00709#:~:text=Alkaline%20Water%20Electrolysis%20(AWE),-ARTICLE%20SECTIONS&text=Electrochemical%20water%20splitting%20consists%20of,oxygen%20evolution%20reaction%20(OER).
        #^ section 2.1
        #https://doi.org/10.1016/j.rser.2015.08.044
        #https://www.energy.gov/eere/fuelcells/hydrogen-production-electrolysis
        #^ "transport of hydroxide ions (OH-) through the electrolyte from the cathode to the anode with hydrogen being generated on the cathode side"
        liters_H2O_pr_Nm3_H2 = 1 # 1L H2O / Nm^3-H2 from water purification.
        #11.126 Nm^3 / 1 kg H2
        h2_production_Nm3 = H2_production_kg*11.126 #Nm^3
        
        liters_of_water_usage = h2_production_Nm3*liters_H2O_pr_Nm3_H2

        # h2_production_mol = H2_production_kg*1e3/(self.M_H2) #mol-H2/dt
        # water_used_for_splitting_kg = h2_production_mol*self.M_H2O/1e3 #kg-H2O
        
        ## cathode: 2H2O + 2e- -> H2 + 2OH-
        
        # 3.7g of KOH per kg H2 (unsure if ratio refers to solvent or solution)
        # 1L H2O / Nm^3-H2 from water purification.
        KOH_to_H2_usage = 3.7 #3.7 grams KOH per kg H2
        KOH_grams_used = H2_production_kg*KOH_to_H2_usage
        #11.126 Nm^3 / 1 kg H2

        #OH- anions are oxidized at anode
        self.BOL_design_info.update({"Feedstock Usage: Liters H2O/kg-H2":11.126,"Feedstock Usage: Grams KOH/kg-H2":KOH_to_H2_usage})

    def cell_bubble_rate_coverage(self, T_stack, I_stack):
        """_summary_

        Args:
            T_stack (_type_): _description_
            I_stack (_type_): _description_

        Returns:
            _type_: _description_
        """
        T_k = convert_temperature([T_stack], "C", "K")[0]
        T_amb = convert_temperature([25], "C", "K")[0]
        J_lim = 30  # [A/cm^2] [Vogt,Balzer 2005]
        j = I_stack / self.cell_area  # [A/cm^2] "nominal current density"
        Pv_H20 = np.exp(
            81.6179 - (7699.68 / T_k) - (10.9 * np.log(T_k)) + (T_k * (9.5891 * 1e-3))
        )
        theta = (
            (self.pressure_operating / (self.pressure_operating - Pv_H20))
            * (-97.25 + 182 * (T_k / T_amb) - 84 * ((T_k / T_amb) ** 2))
            * (j / J_lim) ** (0.3)
        )
        epsilon = (2 / 3) * theta  # bulk bubbling
        return theta, epsilon

    def calc_current_density(self, T_stack, I_stack):
        """_summary_

        Args:
            T_stack (_type_): _description_
            I_stack (_type_): _description_

        Returns:
            _type_: _description_
        """
        theta,epsilon = self.cell_bubble_rate_coverage(T_stack, I_stack)
        A_electrode_eff = self.cell_area * (1 - theta)  # [cm^2]
        j = I_stack / A_electrode_eff  # [A/cm^2]
        return j
    # ---------------------------------- #
    # ----- CELL VOLTAGE EQUATIONS ----- #
    # ---------------------------------- #
    def cell_design(self,T_stack,I_stack):
        
        V_rev = self.cell_reversible_overpotential(T_stack, self.pressure_operating)
        V_act_a, V_act_c = self.cell_activation_overpotential(T_stack, I_stack)
        V_ohm = self.cell_ohmic_overpotential(T_stack, I_stack)
        V_cell = V_rev + V_ohm + V_act_a + V_act_c  # Eqn 4

        V_cell = np.nan_to_num(V_cell)
        return V_cell

    def cell_reversible_overpotential(self, T_stack, P):
        T_K = convert_temperature([T_stack], "C", "K")[0]
        # Eqn 17
        a = (
            (-0.0151 * self.m)
            - (1.6788 * (1e-3) * (self.m**2))
            + (2.2588 * (1e-5) * (self.m**3))
        )
        # Eqn 18 
        b = (
            1
            - ((1.26062 * (1e-3)) * self.m)
            + ((5.6024 * 1e-4) * (self.m**2))
            - ((self.m**3) * (7.8228 * 1e-6))
        )
        # Pv_H20 is vapor pressure of pure water [bar]
        # Eqn 19
        Pv_H20 = np.exp(
            81.6179 - (7699.68 / T_K) - (10.9 * np.log(T_K)) + (T_K * (9.5891 * 1e-3))
        )
        # alpha_h20: water activity of electrolyte solution based on molality [mol/kg]
        # valid for molality ranging from 2-18 mol/kg
        # Eqn 20
        alpha_h20 = np.exp(
            (-0.05192 * self.m)
            + (0.003302 * (self.m**2))
            + (((3.177 * self.m) - (2.131 * self.m**2)) / T_K)
        )
        # Pv_KOH: [bar] vapor pressure of KOH solution
        # Eqn 16
        Pv_KOH = np.exp((2.302 * a) + (b * np.log(Pv_H20)))

        Urev0 = self.cell_Urev0()

        # Eqn 14
        U_rev = Urev0 + ((R * T_K) / (2 * self.F)) * np.log(
            ((P - Pv_KOH) ** 1.5) / alpha_h20
        )
        return U_rev

    def cell_Urev0(self):
        return self.gibbs / (2 * self.F)
    def cell_Utn(self):
        #change in enthalpy (H) over zF
        #h = 285.83 kJ/mol
        enthalpy = 286e3 #kJ/mol
        Utn = enthalpy/(self.z*self.F)
        return Utn
    def cell_activation_overpotential(self, T_stack, I_stack):
        # validated against Figure 5 of Reference
        j_eff = self.calc_current_density(T_stack, I_stack)
        ja = j_eff  # [A/cm^2]
        jc = j_eff  # [A/cm^2]
        T_anode = convert_temperature([T_stack], "C", "K")[0]
        T_cathode = convert_temperature([T_stack], "C", "K")[0]
        # Eqn 14 anode charge transfer coeff
        alpha_a = 0.0675 + 0.00095 * T_anode
        # Eqn 15 cathode charge transfer coeff
        alpha_c = 0.1175 + 0.00095 * T_cathode

        # Table 1
        delta_Ga = 41500  # [J/mol*K]
        delta_Gc = 23450  # [J/mol*K]
        jref_0a = 1.34535 * 10 ** (-5)  # [A/cm^2]
        jref_0c = 1.8456 * 10 ** (-3)  # [A/cm^2]
        Tref = convert_temperature([25], "C", "K")[0]
        gamma_a = 1.25  # anode roughness factor
        gamma_c = 1.05  # cathode roughness factor
        # Eqn 16
        j0c = (
            gamma_c
            * jref_0c
            * np.exp((-1 * delta_Gc / R) * ((1 / T_cathode) - (1 / Tref)))
        )
        # Eqn 16
        j0a = (
            gamma_a
            * jref_0a
            * np.exp((-1 * delta_Ga / R) * ((1 / T_anode) - (1 / Tref)))
        )
        # Eqn 13 - Tafel slope for anode
        ba = (R * T_anode) / (self.z * self.F * alpha_a)
        # Eqn 13 - Tafel slope for cathode
        bc = (R * T_anode) / (self.z * self.F * alpha_c)
        
        if isinstance(I_stack,(float,int)):
            
            if j_eff>0:
                # Eqn 11 - anode activation energy
                V_act_a = ba * np.maximum(0, np.log(ja / j0a))
                # Eqn 12 - cathode activation energy
                V_act_c = bc * np.maximum(0, np.log(jc / j0c))
            else:
                V_act_a = 0
                V_act_c = 0
        else:
            if any(j==0 for j in j_eff):
                # Eqn 11 - anode activation energy
                V_act_a = np.zeros(len(I_stack))
                V_act_c = np.zeros(len(I_stack))
                i_on = np.argwhere(j_eff>0)[:,0]
                # np.log(j_eff[np.argwhere(j_eff>0)[:,0]]/j0a)
                V_act_a[i_on] = ba * np.log(ja[i_on] / j0a)
                # Eqn 12 - cathode activation energy
                V_act_c[i_on] = bc * np.log(jc[i_on] / j0c)
                # V_act_c = np.where(j_eff>0,bc * np.log(jc / j0c),0)
            else:
                V_act_a = ba * np.maximum(0, np.log(ja / j0a))
                V_act_c = bc * np.maximum(0, np.log(jc / j0c))

        return V_act_a, V_act_c

    def cell_ohmic_overpotential(self, T_stack, I_stack):
        R_tot = self.cell_total_resistance(T_stack, I_stack)  # Ohms
        V_ohm = I_stack * R_tot  # [V/cell]
        return V_ohm

    def cell_total_resistance(self, T_stack, I_stack):
        R_a, R_c = self.cell_electrode_resistance(T_stack)
        R_electrode = R_a + R_c
        R_ele_bf, R_ele_b = self.cell_electrolyte_resistance(T_stack, I_stack)  # [Ohms]
        R_electrolyte = R_ele_bf + R_ele_b
        R_membrane = self.cell_membrane_resistance(T_stack)  # [Ohms] VERIFIED for Ohm*cm^2
        R_tot = R_electrode + R_electrolyte + R_membrane  # Ohm

        return R_tot
        
    def cell_electrolyte_resistance(self,T_stack, I_stack):
        T_K = convert_temperature([T_stack], "C", "K")[0]
        sigma_bf = (
            -204.1 * self.M
            - 0.28 * self.M**2
            + 0.5332 * self.M * T_K
            + 20720 * (self.M / T_K)
            + 0.1043 * self.M**3
            - 0.00003 * (self.M**2 * T_K**2)
        )
        R_ele_bf = (100 / sigma_bf) * (
            (self.d_em / self.cell_area) + (self.d_em / self.cell_area)
        )
        theta,epsilon = self.cell_bubble_rate_coverage(T_stack, I_stack)
        R_ele_b = R_ele_bf * ((1 / ((1 - epsilon) ** (3 / 2))) - 1)

        return R_ele_bf, R_ele_b  # Ohms

    def cell_membrane_resistance(self, T_stack):
        Rmem = (0.06 + 80 * np.exp(T_stack / 50)) / (
            10000 * self.cell_area
        )  # Equation 36 - Ohms
        return Rmem

    def cell_electrode_resistance(self,T_stack):
        """_summary_

        Args:
            T_stack (_type_): _description_

        Returns:
            Ra (_type_): _description_
            Rc (_type_): _description_
        """
        tref = 25
        temp_coeff = 0.00586  # 1/degC
        # resistivity of 100% dense electrode at tref
        rho_nickle_0 = 6.4 * 10 ** (-6)  # [Ohm*cm]
        # porosity of electrode
        epsilon_Ni = 0.3
        # Eqn 21 - effective resistance of electrode
        rho_nickle_eff = rho_nickle_0 / ((1 - epsilon_Ni) ** 1.5)
        Ra = (
            rho_nickle_eff
            * (self.e_e / self.cell_area)
            * (1 + (temp_coeff * (T_stack - tref)))
        )
        Rc = (
            rho_nickle_eff
            * (self.e_e / self.cell_area)
            * (1 + (temp_coeff * (T_stack - tref)))
        )

        return Ra, Rc  
    
    # -------------------------------------- #
    # ----- CELL DEGRADATION EQUATIONS ----- #
    # -------------------------------------- #
    def cell_degradation(self,V_cell,cluster_status):
        if self.include_degradation_penalty:

            V_cell = V_cell*cluster_status
            
            V_deg_uptime = self.cell_steady_degradation(V_cell,cluster_status)
            V_deg_onoff = self.cell_onoff_degradation(cluster_status)
            V_fatigue = self.cell_fatigue_degradation(V_cell)

            V_deg = np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff) + V_fatigue
            self.simulation_results.update({"Simulation Final Steady Deg [V/cell]":np.cumsum(V_deg_uptime)[-1]})
            self.simulation_results.update({"Simulation Final On/Off Deg [V/cell]":np.cumsum(V_deg_onoff)[-1]})
            self.simulation_results.update({"Simulation Final Fatigue Deg [V/cell]":V_fatigue[-1]})
        else:
            V_deg = np.zeros(len(V_cell))
            self.simulation_results.update({"Simulation Final Steady Deg [V/cell]":0})
            self.simulation_results.update({"Simulation Final On/Off Deg [V/cell]":0})
            self.simulation_results.update({"Simulation Final Fatigue Deg [V/cell]":0})
        return V_deg

    def cell_fatigue_degradation(self,V_cell,dt_fatigue_calc_hrs=168):
        V_fatigue_ts=np.zeros(len(V_cell))
        lifetime_fatigue_deg = 0 
        if np.max(V_cell)!=np.min(V_cell):
            
            rf_cycles = rainflow.count_cycles(V_cell, nbins=10)
            rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
            lifetime_fatigue_deg=rf_sum*self.rate_fatigue

            rf_track=0
            t_calc=np.arange(0,len(V_cell)+dt_fatigue_calc_hrs,dt_fatigue_calc_hrs) 
            for i in range(len(t_calc)-1):
                voltage_signal_temp = V_cell[np.nonzero(V_cell[t_calc[i]:t_calc[i+1]])]
                if np.size(voltage_signal_temp)==0:
                    rf_sum = 0
                elif np.max(voltage_signal_temp)==np.min(voltage_signal_temp):
                    rf_sum=0
                else:
                    rf_cycles=rainflow.count_cycles(voltage_signal_temp, nbins=10)
                    rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
                rf_track+=rf_sum
                V_fatigue_ts[t_calc[i]:t_calc[i+1]]=rf_track*self.rate_fatigue

        self.simulation_results.update({"Lifetime Fatigue Deg [V/sim]":lifetime_fatigue_deg})
        return V_fatigue_ts

    def cell_steady_degradation(self,V_cell,cluster_status):
        steady_deg_per_hr=self.dt*self.steady_deg_rate*V_cell*cluster_status
        # cumulative_Vdeg=np.cumsum(steady_deg_per_hr)
        # self.steady_deg_rate
        self.simulation_results.update({"On-time/sim [hrs]":np.sum(cluster_status)*(self.dt/3600)})
        return steady_deg_per_hr

    def cell_onoff_degradation(self,cluster_status):
        change_stack=np.diff(cluster_status)
        cycle_cnt = np.where(change_stack < 0, -1*change_stack, 0)
        
        cycle_cnt = np.array([0] + list(cycle_cnt))
        self.off_cycle_cnt = np.sum(cycle_cnt)
        stack_off_deg_per_hr= self.onoff_deg_rate*cycle_cnt
        self.simulation_results.update({"Off-cycles/sim":self.off_cycle_cnt})
        return stack_off_deg_per_hr
    
    
    # --------------------------------- #
    # ----- CELL OUTPUT EQUATIONS ----- #
    # --------------------------------- #
    def calc_faradaic_efficiency(self,T_stack,I_stack):

        f1 = 250  # [mA^2/cm^4]
        f2 = 0.96  # [-]

        j = self.calc_current_density(T_stack, I_stack)  # [A/cm^2]
        j *= 1000  # [mA/cm^2]
        #Faradaic Efficiency
        eta_F = f2 * (j**2) / (f1 + j**2)
        return eta_F

    def cell_H2_production_rate(self,T_stack,I_stack):
        eta_F = self.calc_faradaic_efficiency(T_stack, I_stack)
        h2_prod_mol = eta_F * I_stack / (2 * self.F) #[mol/sec]
        mfr = self.M_H2 * h2_prod_mol  # [g/sec]
        mfr_H2 = self.dt*mfr / 1e3  # [kg/cell-dt]
        return mfr_H2

    def cell_O2_production_rate(self,T_stack,I_stack):
        eta_F = self.calc_faradaic_efficiency(T_stack, I_stack)
        o2_prod_mol = eta_F * I_stack / (4 * self.F) #[mol/sec]
        mfr = self.M_O2 * o2_prod_mol  # [g/sec]
        mfr_O2 = self.dt*mfr / 1e3  # [kg/cell-dt]
        return mfr_O2
    
    def stack_reverse_faradays(self,H2_required_per_stack_kg):
        #NOTE: Runtime warning when n_f=0
        I_reqd_BOL_noFaradaicLoss=(H2_required_per_stack_kg*1000*2*self.F)/(1*self.n_cells*self.dt*self.M_H2)
        n_f=self.calc_faradaic_efficiency(self.T_stack,I_reqd_BOL_noFaradaicLoss)
        I_reqd=(H2_required_per_stack_kg*1000*2*self.F)/(n_f*self.n_cells*self.dt*self.M_H2)
        return np.nan_to_num(I_reqd)

    def estimate_cell_degradation_from_demand(self,H2_required_cluster_kg):

        H2_required_per_stack_kg = H2_required_cluster_kg/self.n_stacks
        I_reqd = self.stack_reverse_faradays(H2_required_per_stack_kg)
        V_reqd = self.cell_design(self.T_stack,I_reqd)
        if isinstance(H2_required_cluster_kg,float):
            #steady
            V_deg_per_hr=self.steady_deg_rate*V_reqd*self.dt
            V_deg=np.arange(0,self.d_eol+V_deg_per_hr,V_deg_per_hr)
            
        else:
            cluster_status = self.calc_cluster_status(I_reqd)
            V_deg = self.cell_degradation(V_reqd,cluster_status)
            #TODO: only return V-deg until EOL
            if np.max(V_deg)>self.d_eol:
                idx_dead = np.argwhere(V_deg>self.d_eol)[0][0]
                V_deg = V_deg[0:idx_dead]

        # P_reqd_per_hr_stack=I_reqd*(V_reqd + V_deg)*self.n_cells/1000 #kW
        # P_required_per_hr_system=self.n_stacks*P_reqd_per_hr_stack #kW
            
        return V_deg
        
    # ------------------------------------ #      
    # ----- ANALYSIS/POST-PROCESSING ----- #
    # ------------------------------------ #     
if __name__ == "__main__":
    
    alk = ALK_Clusters(cluster_size_mw=1,plant_life=30)
    from greenheart.simulation.technologies.hydrogen.electrolysis.ALK_electrolyzer_tools import get_efficiency_curve
    df = get_efficiency_curve(alk,file_desc = "July2024")
    # from greenheart.simulation.technologies.hydrogen.electrolysis.ALK_electrolyzer_tools import plot_IV_curve
    # plot_IV_curve(alk,file_desc="7-bar")
    alk.nominal_current*alk.V_cell_nominal
    alk.V_cell_nominal #around 1.9
    alk.nominal_current_density #0.5
    # A_cell = np.arange(300,2500,100)
    # J_cell_nom = np.arange(0.25,0.55,0.05)
    # V_cells = np.zeros(len(J_cell_nom))
    # n_cells = np.zeros(len(J_cell_nom))
    # T_stack = 60
    # for i,j in enumerate(J_cell_nom):
    #     alk.cell_area = 1500
    #     alk.pressure_operating = 1
    #     alk.nominal_current_density = j 
    #     alk.system_design(T_stack)
    #     V_cells[i] = alk.V_cell_nominal
    #     # n_cells = 1000/(alk.nominal_current*alk.V_cell_nominal/1e3)
    #     n_cells[i] = 1e6/(alk.nominal_current*alk.V_cell_nominal)
    []

