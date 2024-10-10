import math
import numpy as np
import sys
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import rainflow
from scipy import interpolate
from scipy.constants import atm, mmHg, bar
from scipy.constants import R, physical_constants, convert_temperature

def stack_power_to_current(P_T,p1,p2,p3,p4,p5,p6): #calculates i-v curve coefficients given the stack power and stack temp
    pwr,tempc=P_T
    # i_stack=p1*(pwr**2) + p2*(tempc**2)+ (p3*pwr*tempc) +  (p4*pwr) + (p5*tempc) + (p6)
    i_stack=p1*(pwr**3) + p2*(pwr**2) +  (p3*pwr) + (p4*pwr**(1/2)) + p5
    return i_stack 

class PEM_Clusters:
    # num hydrogen molecules transferred per reaction
    z: int = 2 #TODO: change to z_c
    F: float = 96485.34  # Faraday's Constant (C/mol) or [As/mol]
    R: float = 8.314  # Ideal Gas Constant (J/mol/K)

    M_H: float = 1.00784  # molecular weight of Hydrogen [g/mol]
    M_H2: float = 2.016 #[g/mol]
    M_O: float = 15.999  # molecular weight of Oxygen [g/mol]
    M_O2: float = 31.999 #[g/mol]
    
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
            uptime_hours_until_eol = 77600, 
            n_cycles_until_eol = 614, #update to PEM default
            ramp_rate_percent = 0.99, #update to PEM default
            turndown_ratio = 0.10, 
            cold_start_delay = 600,
            anode_pressure_bar = 1.01325,
            cathode_pressure_bar = 1.01325,
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
        self.nominal_current_density = 2.0 #[A/cm^2]
        
        # OPERATING CONDITIONS
        ##TODO: update pressure_operating to be PEM-specific
        self.anode_pressure = anode_pressure_bar #[bar] operating pressure at anode
        self.cathode_pressure = cathode_pressure_bar #[bar] operating pressure at cathode
        T_stack = 80 #Celsius

        # CLUSTER DESIGN PARAMETERS
        #ASSUMES THAT STACK IS 1 MW THEREFORE n_stacks = cluster_size_mw
        self.n_stacks = cluster_size_mw

        # STACK DESIGN PARAMETERS
        self.stack_rating_kW = 1000  # 1 MW - this is reset in system_design
        self.n_cells = 130

        # CELL DESIGN PARAMETERS
        self.cell_area = 1920 # [cm^2] membrane and electrode area
        self.e_m = 0.018 #[cm] membrane thickness - check if used

        # CELL DEGRADATION RATES
        self.onoff_deg_rate=1.47821515e-04 #[V/off-cycle]
        self.rate_fatigue = 3.33330244e-07 #multiply by rf_track

        # INITIALIZATION
        self.initalize_outputs(plant_life,run_LTA,debug_mode)
        self.feedstock_usage()
        self.system_design(T_stack)
        self.curve_coeff=self.create_power_current_curve(T_stack)

        self.eol_eff_drop = eol_eff_percent_loss/100
        self.d_eol=self.find_eol_voltage_val(eol_eff_percent_loss)
        self.BOL_design_info.update({"EOL Cell Voltage Degradation Value [V/cell]":self.d_eol})
        # CELL DEGRADATION RATES
        self.steady_deg_rate = self.reset_uptime_degradation_rate(uptime_hours_until_eol)
        # self.onoff_deg_rate = self.reset_on_off_degradation_rate(n_cycles_until_eol)
        self.describe_degradation_rates()
        
    
    # def reset_on_off_degradation_rate(self,n_cycles_until_eol):
    #     onoff_deg_rate = self.d_eol/n_cycles_until_eol
    #     return onoff_deg_rate
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

        cell_nominal_h2_kg = self.cell_H2_production_rate(self.nominal_current)
        self.stack_nominal_h2_kg = cell_nominal_h2_kg*self.n_cells #[kg H2/dt-stack]
        self.cluster_nominal_h2_kg = self.stack_nominal_h2_kg*self.n_stacks #[kg H2/dt-cluster]

        cell_min_h2_kg = self.cell_H2_production_rate(self.min_current)
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
        self.BOL_design_info.update({"Stack Anode Pressure [bar]":self.anode_pressure})
        self.BOL_design_info.update({"Stack Cathode Pressure [bar]":self.cathode_pressure})
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
            else:
                self.simulation_results.update({"--Total {}".format(key):sum(value)})
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
        hydrogen_produced_kg_nom = self.cell_H2_production_rate(I_stack)*self.n_cells*self.n_stacks
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
        #TODO: add
        pass
    def calculate_efficiency(self,hydrogen_produced,power_consumed):
        return power_consumed/hydrogen_produced
        
    def run_LTA_analysis(self,V_deg,V_cell_nom,I_stack_nom):
        from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_LTA import PEM_LTA
        lta = PEM_LTA(self)
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
        power_reqd_kW = (I_reqd*(V_reqd + V_deg_est)*self.n_cells*self.n_stacks)/1e3
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
                current_density[idx] = self.calc_current_density(current_range[i])
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

    def feedstock_usage(self):
        
        self.BOL_design_info.update({"Feedstock Usage: Liters H2O/kg-H2":10})
    
    def calc_current_density(self, I_stack):
        """_summary_

        Args:
            T_stack (_type_): _description_
            I_stack (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        j = I_stack / self.cell_area  # [A/cm^2]
        return j
    # ---------------------------------- #
    # ----- CELL VOLTAGE EQUATIONS ----- #
    # ---------------------------------- #
    def cell_design(self,T_stack,I_stack):
        
        V_rev = self.cell_reversible_overpotential(T_stack)
        V_act_a, V_act_c = self.cell_activation_overpotential(T_stack, I_stack)
        V_ohm = self.cell_ohmic_overpotential(T_stack, I_stack)
        V_cell = V_rev + V_ohm + V_act_a + V_act_c  # Eqn 4

        V_cell = np.nan_to_num(V_cell)
        return V_cell
    def antoine_formula(self,T_stack):
        A = 8.07131
        B = 1730.63
        C = 233.426
        p_H2O_sat_mmHg = 10 ** (A - (B / (C + T_stack))) 
        #convert mmHg to atm
        mmHg_2_atm = mmHg/atm
        p_H2O_sat_atm=p_H2O_sat_mmHg*mmHg_2_atm  
        return p_H2O_sat_atm

    def arden_buck(self,T_stack):
        p_h2O_sat_kPa = (0.61121* np.exp((18.678 - (T_stack / 234.5)) * (T_stack / (257.14 + T_stack))))
        p_H2O_sat_atm = (p_h2O_sat_kPa*1e3)*atm
        # p_H2O_bar = p_h2O_sat_kPa*0.01
        return p_H2O_sat_atm
    def cell_calc_Urev(self,T_stack):
        #UNUSED RIGHT NOW BUT SHOULD REPLACE cell_reversible_overpotential
        T_K = convert_temperature([T_stack], "C", "K")[0]
        Urev0 = self.cell_Urev0()
        p_H2O_sat_atm = self.arden_buck(T_stack)
        p_H2O_sat_bar = p_H2O_sat_atm*(bar/atm)
        p_H2 = self.cathode_pressure - p_H2O_sat_bar
        p_O2 = self.anode_pressure - p_H2O_sat_bar
        b = (p_H2*np.sqrt(p_O2))/p_H2O_sat_bar #maybe should be in Pa?
        U_rev = Urev0 + ((self.R*T_K)/(2*self.F))*(np.log(b))
        return U_rev


    def cell_reversible_overpotential(self, T_stack):
        # updated for PEM
        #TODO: make pressure an attribute
       
        T_K = convert_temperature([T_stack], "C", "K")[0]
        Urev0 = self.cell_Urev0()
        panode_atm = self.anode_pressure*(atm/bar) #[atm] total pressure at the anode
        pcathode_atm = self.anode_pressure*(atm/bar) #[atm] total pressure at the cathode
        #TODO: add in daltons law of partial pressures
        patmo_atm = 1 #atmospheric pressure
        #TODO: replace Antoine formula with Arden-Buck
        p_H2O_sat_atm = self.antoine_formula(T_stack)
        
        
        U_rev = Urev0 + ((self.R*T_K)/(2*self.F))*(np.log(((panode_atm-p_H2O_sat_atm)/patmo_atm)*np.sqrt((pcathode_atm-p_H2O_sat_atm)/patmo_atm))) 
        
        return U_rev

    def cell_Urev0(self):
        #http://dx.doi.org/10.1016/j.ijhydene.2017.03.046
        # Urev0 = (self.gibbs / (2 * self.F)) - (0.9*1e-3)*(T_K-298)
        return self.gibbs / (2 * self.F)
    def cell_Utn(self):
        #change in enthalpy (H) over zF
        #h = 285.83 kJ/mol
        enthalpy = 286e3 #kJ/mol
        Utn = enthalpy/(self.z*self.F)
        return Utn
    def cell_activation_overpotential(self, T_stack, I_stack):
        # updated for PEM
        # validated against Figure 5 of Reference
        T_K = convert_temperature([T_stack], "C", "K")[0]
        #current density [A/cm^2]
        i = self.calc_current_density(I_stack)
        # Anode charge transfer coefficient
        a_a = 2  
        # Cathode charge transfer coefficient
        a_c = 0.5  
        #anode exchange current density
        i_o_a = 2 * (10 ** (-7)) 
        #cathode exchange current density
        i_o_c = 2 * (10 ** (-3)) 
        V_act_a = (((self.R * T_K) / (a_a * self.F)) * np.arcsinh(i / (2 * i_o_a)))
        V_act_c= (((self.R * T_K) / (a_c * self.F)) * np.arcsinh(i / (2 * i_o_c)))
        return V_act_a, V_act_c

    def cell_ohmic_overpotential(self, T_stack, I_stack):
        # updated for PEM
        R_tot = self.cell_total_resistance(T_stack)  # Ohms
        i = self.calc_current_density(I_stack)
        V_ohm = i * R_tot  # [V/cell]
        return V_ohm

    def cell_total_resistance(self, T_stack):
        # updated for PEM
        R_electrode = self.cell_electrode_resistance()
        R_membrane = self.cell_membrane_resistance(T_stack)  # [Ohms] VERIFIED for Ohm*cm^2
        R_tot = R_electrode + R_membrane  # Ohm*cm^2

        return R_tot
        
    

    def cell_membrane_resistance(self, T_stack):
        T_K=T_stack+ 273.15 
        lambda_water_content = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        # membrane proton conductivity [S/cm]
        sigma = ((0.005139 * lambda_water_content) - 0.00326) * np.exp(
            1268 * ((1 / 303) - (1 / T_K)))   
        #ionic resistance [ohms*cm^2]
        R_cell = (self.e_m / sigma) 
        return R_cell

    def cell_electrode_resistance(self):
        # [ohms*cm^2] from Table 1 in  https://journals.utm.my/jurnalteknologi/article/view/5213/3557
        R_elec=3.5*(10 ** (-5))
        return R_elec 
    
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
    def calc_faradaic_efficiency(self,I_stack):
        # updated for PEM
        f1 = 250  # [mA^2/cm^4]
        f2 = 0.996  # [-]

        j = self.calc_current_density(I_stack)  # [A/cm^2]
        j *= 1000  # [mA/cm^2]
        #Faradaic Efficiency
        eta_F = f2 * (j**2) / (f1 + j**2)
        return eta_F

    def cell_H2_production_rate(self,I_stack):
        eta_F = self.calc_faradaic_efficiency(I_stack)
        h2_prod_mol = eta_F * I_stack / (2 * self.F) #[mol/sec]
        mfr = self.M_H2 * h2_prod_mol  # [g/sec]
        mfr_H2 = self.dt*mfr / 1e3  # [kg/cell-dt]
        return mfr_H2

    def cell_O2_production_rate(self,I_stack):
        eta_F = self.calc_faradaic_efficiency(I_stack)
        o2_prod_mol = eta_F * I_stack / (4 * self.F) #[mol/sec]
        mfr = self.M_O2 * o2_prod_mol  # [g/sec]
        mfr_O2 = self.dt*mfr / 1e3  # [kg/cell-dt]
        return mfr_O2
    
    def stack_reverse_faradays(self,H2_required_per_stack_kg):
        #NOTE: Runtime warning when n_f=0
        I_reqd_BOL_noFaradaicLoss=(H2_required_per_stack_kg*1000*2*self.F)/(1*self.n_cells*self.dt*self.M_H2)
        n_f=self.calc_faradaic_efficiency(I_reqd_BOL_noFaradaicLoss)
        # I_reqd = np.where(n_f>0,(H2_required_per_stack_kg*1000*2*self.F)/(n_f*self.n_cells*self.dt*self.M_H2),0)
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


