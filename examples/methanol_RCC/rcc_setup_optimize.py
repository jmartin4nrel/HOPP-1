import sys
import os
import hybrid
import copy
from dotenv import load_dotenv
from math import sin, pi
import PySAM.Singleowner as so
import pandas as pd
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.run_h2_PEM import run_h2_PEM
import numpy as np
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
from tools.analysis import create_cost_calculator
import json

import warnings
warnings.filterwarnings("ignore")

def setup_power_calcs(solar_size_kw):
    """
    A function to facilitate plant setup for POWER calculations, assuming one wind turbine.
    
    INPUT VARIABLES
    scenario: dict, the H2 scenario of interest
    solar_size_mw: float, the amount of solar capacity in MW
    storage_size_mwh: float, the amount of battery storate capacity in MWh
    storage_size_mw: float, the amount of battery storate capacity in MW
    interconnection_size_mw: float, the interconnection size in MW

    OUTPUTS
    hybrid_plant: the hybrid plant object from HOPP for power calculations
    """

    # Set API key
    load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

    # Step 1: Establish output structure and special inputs
    year = 2013
    sample_site['year'] = year
    elyzer_size_kw = 384780
    iconn_size_kw = 668101

    site = SiteInfo(sample_site, hub_height=135)
    
    technologies = {'pv':
                        {'system_capacity_kw': solar_size_kw},
                    'wind':
                        {'num_turbines': 100,
                            'turbine_rating_kw': 7000,
                            'hub_height': 135,
                            'rotor_diameter': 200},
                    'grid': elyzer_size_kw,
                    }
    
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=iconn_size_kw)

    return hybrid_plant


def setup_cost_calcs(scenario,hybrid_plant,electrolyzer_size_mw,wind_size_mw,solar_size_mw,
                    solar_cost_multiplier=1.0):
    """
    A function to facilitate plant setup for COST calculations. 
    
    INPUT VARIABLES
    scenario: dict, the H2 scenario of interest
    hybrid_plant: the hybrid plant object from setup_power_calcs, so we don't need to reinput everything
    electrolyzer_size_mw: float, the exlectrolyzer capacity in MW
    wind_size_mw: float, the amount of wind capacity in MW
    solar_size_mw: float, the amount of solar capacity in MW
    solar_cost_multiplier: float, if you want to test design sensitivities to solar costs. Multiplies the solar costs

    OUTPUTS
    hybrid_plant: the hybrid plant object from HOPP for cost calculations
    """

    # Step 1: Establish output structure and special inputs
    year = 2013
    sample_site['year'] = year
    useful_life = 30

    sample_site['lat'] = scenario['Lat']
    sample_site['lon'] = scenario['Long']
    wind_cost_kw = scenario['Wind Cost KW']
    solar_cost_kw = scenario['Solar Cost KW']*solar_cost_multiplier
    storage_cost_kw = scenario['Storage Cost KW']
    storage_cost_kwh = scenario['Storage Cost KWh']

    #Todo: Add useful life to .csv scenario input instead
    scenario['Useful Life'] = useful_life

    interconnection_size_mw = electrolyzer_size_mw

    hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                              bos_cost_source='CostPerMW',
                                                              wind_installed_cost_mw=wind_cost_kw * 1000,
                                                              solar_installed_cost_mw=solar_cost_kw * 1000,
                                                              storage_installed_cost_mw=storage_cost_kw * 1000,
                                                              storage_installed_cost_mwh=storage_cost_kwh * 1000
                                                              ))
    hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.33   
    if solar_size_mw > 0:
        hybrid_plant.pv._financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hybrid_plant.pv._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        if scenario['ITC Available']:
            hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 26
        else:
            hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 0

    if wind_size_mw > 0:
        hybrid_plant.wind._financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hybrid_plant.wind._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        if scenario['PTC Available']:
            ptc_val = 0.022
        else:
            ptc_val = 0.0

        interim_list = list(
            hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount)
        interim_list[0] = ptc_val
        hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
        hybrid_plant.wind._system_model.Turbine.wind_turbine_hub_ht = scenario['Tower Height']

    hybrid_plant.ppa_price = 0.05
    hybrid_plant.wind.system_capacity_kw = wind_size_mw*1000
    return hybrid_plant


def calculate_lcoe_rcc_optimizer(solar_capacity_mw):
    """
    A function to calculate lcoe within an optimization. The wind generation is calculated for a single turbine, then scaled up to the entire capacity.
    This means that no wake losses are captured, but allows the power generation to be continuous with respect to wind capacity. The h_lcoe is not actually 
    continuous at this point, because the costs assume integer values of turbines. This will need to be improved for gradient-based optimization.
    
    INPUT VARIABLES
    bat_model: the battery model to be used. See example scripts.
    electrolyzer_size_mw: float, the exlectrolyzer capacity in MW
    wind_size_mw: float, the amount of wind capacity in MW
    solar_size_mw: float, the amount of solar capacity in MW
    battery_storage_mwh: float, the amount of battery storage in MWh
    battery_charge_rate: float, the battery charge rate in MW
    battery_discharge_rate: float, the battery discharge rate in MW
    scenario: dict, the H2 scenario of interest
    buy_from_grid: Bool, can the plant buy from the grid
    sell_to_grid: Bool, can the plant sell to the grid
    solar_cost_multiplier: float, if you want to test design sensitivities to solar costs. Multiplies the solar costs
    interconnection_size_mw: interconnection size in MW. False if not grid connected


    OUTPUTS
    h_lcoe: float, the levelized cost of hydrogen
    aep: float, the total energy production from the wind and solar
    h2_output: float, the total annual output of hydrogen
    total_system_installed_cost: float, the total system installed costs
    total_annual_operating_costs: float, the total annual operating costs
    electrolyzer_cf: float, the electrolyzer capacity factor
    """

    elyzer_size_kw = 384780
    elyzer_cf = 0.97
        
    solar_capacity_kw = solar_capacity_mw*1000

    hybrid_plant = setup_power_calcs(solar_capacity_kw) 

    useful_life = 30

    hybrid_plant.simulate(useful_life)
    gen_kw = list(hybrid_plant.generation_profile.hybrid[0:8760])
    lcoe = hybrid_plant.lcoe_real.hybrid
    orig_lcoe = copy.copy(lcoe)
    orig_total_cost = sum(copy.deepcopy(gen_kw))*orig_lcoe

    # ppa_price_kwh = 0.04 
    sell_price = 0.01#ppa_price_kwh
    buy_price = 0.05#ppa_price_kwh

    # Find excess generation above electrolyzer capcity and sell to grid
    profit_from_selling_to_grid = 0.0
    excess_energy = [0]*8760    
    for i in range(len(gen_kw)):
        if gen_kw[i] > elyzer_size_kw:
            excess_energy[i] = (gen_kw[i]-elyzer_size_kw)
            profit_from_selling_to_grid += (gen_kw[i]-elyzer_size_kw)*sell_price
            gen_kw[i] = elyzer_size_kw
    
    # Buy grid electricity to meet electrolyzer capacity factor
    cost_to_buy_from_grid = 0.0
    purchase_needed = (elyzer_size_kw*elyzer_cf-np.mean(gen_kw))*8760
    shortfall = np.subtract(elyzer_size_kw,gen_kw)
    shortfall_inds = np.flip(np.argsort(shortfall))
    diff_shortfall = -np.diff(shortfall[shortfall_inds])
    shortfall_changes = np.squeeze(np.argwhere(diff_shortfall))
    purchase = np.zeros(8760)
    shortfall_change = 1
    while np.sum(purchase) < purchase_needed and shortfall_change < len(shortfall_changes):
        purchase[shortfall_inds[:(1+shortfall_changes[shortfall_change])]] += diff_shortfall[shortfall_changes[shortfall_change-1]]
        shortfall_change += 1
    for i in range(len(gen_kw)):
        cost_to_buy_from_grid += purchase[i]*buy_price
        gen_kw[i] += purchase[i]

    # Adjust lcoe
    new_total_cost = orig_total_cost+cost_to_buy_from_grid-profit_from_selling_to_grid
    lcoe = new_total_cost/sum(gen_kw)

    return lcoe


if __name__=="__main__":
    bat_model = SimpleDispatch()

    solar_capacity_mw = [100,300,500,700,900]

    N = len(solar_capacity_mw)
    lcoe = np.zeros(N)
    CF = np.zeros(N)

    for i in range(N):
        lcoe[i] = calculate_lcoe_rcc_optimizer(solar_capacity_mw[i])

    print("lcoe: ", lcoe)