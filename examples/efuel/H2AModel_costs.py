import math
import pandas as pd
import numpy as np

'''
jmartin4nrel edits 1/30/23: Fixed to conform with "future-central-pem-electrolysis-version-nov20.xlsm"
(The Excel macro sheet downloaded from the NREL H2A website: https://www.nrel.gov/hydrogen/h2a-production-models.html)

- Eliminated unused (but required) annual production argument (redundant with avg_daily_H2_production) - used to be 3rd argument
- Fixed mix-up of "force_system_size" (boolean) with "forced_system_size" (number, was accidentally being used instead of boolean)
- Changed the default of this boolean to False - default behavior matches spreadsheet
- Before, first two arguments were being used to scale the BASELINE plant, and the "Scaling Factor" section was doing nothing.
    Former references to the first two arguments have been replaced with constants to calculated baseline plant costs, and the
    arguments are now used to calculate Scaling Factors that adjust the (constant) baseline plant costs
- Added startup year input to interpolate between "current" (2015) and "future" (2040) scenarios

Inputs: cap_factor: % capacity that the SCALED H2 plant will be run at
        avg_daily_H2_production: average ACTUAL daily production (kg H2/day) that is needed from the scaled plant
                                 (NOT the design capacity - divide by cap_factor to get design capacity)
        startup_year: Year that plant would start up, to interpolate between model constants in "current" and "future" scenarios
        (optional) scenario: 'Current' or 'Future', whether to interpolate projected cost improvements or not

Returns: Total capex [$],
         Total fixed opex [$/year],
         Electricity use [kWh/kg H2],
         Water use [kg H2O/kg H2]

        ALL BASIS 2016 DOLLARS

Now, running 
    >>> H2AModel(0.97, 0.97*59150, 2040)
will produce the same results as the spreadsheet available online, and will scale costs for a needed ACTUAL daily output
Scaling of costs with different values for the 2nd argument will happen through exponential capex scaling, NOT modifying the model
'''

def H2AModel_costs(cap_factor, avg_daily_H2_production, year, scenario='Future', h2a_for_hopp=True, force_system_size=False,
                    forced_system_size=50, force_electrolyzer_cost=False, forced_electrolyzer_cost_kw=200, useful_life = 30):


    # -----------------------------------------------------LOOKUP TABLE------------------------------------------------------------#

    def lookup_value(scenario, value, year):
    
        lookup_table = {'scenario':                             ['Current', 'Future'],
                        'startup_year':                         [2015,      2040],
                        'current_density':                      [2,         3],
                        'voltage':                              [1.9,       1.8],
                        'H2_outlet_pressure':                   [450,       700],
                        'cell_active_area':                     [450,       1500],
                        'degradation_rate':                     [1.5,       1],
                        'stack_life':                           [7,         10],
                        'stack_degradation_oversize':           [0.13,      0.183],
                        'total_system_electrical_usage':        [55.5,      51.3],
                        'stack_electrical_usage':               [50.4,      47.8],
                        'system_unit_cost':                     [1.3,       0.77],
                        'mechanical_BoP_unit_cost':             [76,        45.84],
                        'electrical_BoP_cost':                  [82,        68],
                        'stack_installation_factor':            [1.12,      1.1],
                        'electrical_BoP_installation_factor':   [1.12,      1.1]}

        min_idx = lookup_table['scenario'].index('Current')
        max_idx = lookup_table['scenario'].index('Future')

        min_year = lookup_table['startup_year'][min_idx]
        max_year = lookup_table['startup_year'][max_idx]
        min_val = lookup_table[value][min_idx]
        max_val = lookup_table[value][max_idx]

        if scenario is 'Current':
            value = min_val
        else:
            if year < min_year:
                value = min_val
            elif year > max_year:
                value = max_val
            else:
                interp_frac = (year-min_year)/(max_year-min_year)
                value = min_val+(max_val-min_val) * interp_frac

        return value
    
    # ---------------------------------------------------H2A PROCESS FLOW----------------------------------------------------------#

    baseline_cap_factor = 0.97
    baseline_daily_H2_production = 50000 # kg H2/day
    '''
    These WERE arguments, but are fixed in the model - the argument should be used to calculate plant SCALING FACTORS instead
    '''

    current_density = lookup_value(scenario, 'current_density', year)  # A/cm^2
    voltage = lookup_value(scenario, 'voltage', year)  # V/cell
    operating_temp = 80  # C
    H2_outlet_pressure = lookup_value(scenario, 'H2_outlet_pressure', year)  # psi
    cell_active_area = lookup_value(scenario, 'cell_active_area', year)  # cm^2
    cellperstack = 150  # cells
    degradation_rate = lookup_value(scenario, 'degradation_rate', year)  # mV/1000 hrs
    stack_life = lookup_value(scenario, 'stack_life', year)  # years
    hours_per_stack_life = stack_life * 365 * 24 * baseline_cap_factor  # hrs/life
    degradation_rate_Vperlife = hours_per_stack_life * degradation_rate / 1000  # V/life
    stack_degradation_oversize = lookup_value(scenario, 'stack_degradation_oversize', year)  # factor
    peak_daily_production_rate = baseline_daily_H2_production * (1 + stack_degradation_oversize)  # kgH2/day
    baseline_plant_output = peak_daily_production_rate # Corresponds with C27 on the original Excel sheet

    total_active_area = math.ceil(
        (baseline_daily_H2_production / 2.02 * 1000 / 24 / 3600) * 2 * 96485 / current_density / (100 ** 2))  # m^2
    total_active_area_degraded = math.ceil(
        (peak_daily_production_rate / 2.02 * 1000 / 24 / 3600) * 2 * 96485 / current_density / (100 ** 2))  # m^2

    stack_electrical_usage = lookup_value(scenario, 'stack_electrical_usage', year)  # kWh/kgH2
    total_system_electrical_usage = lookup_value(scenario, 'total_system_electrical_usage', year)  # kWh/kg H2
    BoP_electrical_usage = total_system_electrical_usage-stack_electrical_usage  # kWh/kgH2
    
    if force_system_size:
        total_system_input = forced_system_size  # MW
        stack_input_power = (stack_electrical_usage/ total_system_electrical_usage) * forced_system_size  # MW
    else:
        total_system_input = total_system_electrical_usage / 24 * peak_daily_production_rate / 1000  # MW
        stack_input_power = stack_electrical_usage / 24 * peak_daily_production_rate / 1000  # MW

    process_water_flowrate = 3.78

    system_unit_cost = lookup_value(scenario, 'system_unit_cost', year) #* 300/342 # $/cm^2
    stack_system_cost = system_unit_cost / (current_density * voltage) * 1000  # $/kW
    mechanical_BoP_unit_cost = lookup_value(scenario, 'mechanical_BoP_unit_cost', year)  # kWhH2/day
    mechanical_BoP_cost = mechanical_BoP_unit_cost * peak_daily_production_rate / stack_input_power / 1000  # $/kW
    electrical_BoP_cost = lookup_value(scenario, 'electrical_BoP_cost', year)  # $/kW
    total_system_cost_perkW = stack_system_cost + mechanical_BoP_cost + electrical_BoP_cost  # $/kW
    total_system_cost_perkW = total_system_cost_perkW
    if force_electrolyzer_cost:
        total_system_cost = forced_electrolyzer_cost_kw * stack_input_power * 1000
    else:
        total_system_cost = total_system_cost_perkW * stack_input_power * 1000  # $


    # -------------------------------------------------CAPITAL COST--------------------------------------------------------------#


    gdpdef = {'Year': [2015, 2016, 2017, 2018, 2019, 2020], 'CEPCI': [104.031, 104.865, 107.010, 109.237, 111.424,
                                                                      113.415]}  # GDPDEF (2012=100), https://fred.stlouisfed.org/series/GDPDEF/
    CEPCI = pd.DataFrame(data=gdpdef)  # Deflator Table

    pci = {'Year': [2015, 2016, 2017, 2018, 2019, 2020],
           'PCI': [556.8, 541.7, 567.5, 603.1, 607.5, 610]}  # plant cost index, Chemical Engineering Magazine
    CPI = pd.DataFrame(data=pci)

    basis_year_for_capital_cost = 2016
    current_year_for_capital_cost = 2016
    CEPCI_inflator = int((CEPCI.loc[CEPCI['Year'] == current_year_for_capital_cost, 'CEPCI'])) / int(
        (CEPCI.loc[CEPCI['Year'] == basis_year_for_capital_cost, 'CEPCI']))
    consumer_price_inflator = int(CPI.loc[CPI['Year'] == current_year_for_capital_cost, 'PCI']) / int(
        CPI.loc[CPI['Year'] == basis_year_for_capital_cost, 'PCI'])  # lookup

    # --------------------------CAPITAL INVESTMENT---------------------------------#
    # ----Inputs required in basis year (2016$)----#

    baseline_uninstalled_stack_capital_cost = CEPCI_inflator * consumer_price_inflator * system_unit_cost * total_active_area_degraded * 100 ** 2  # ($2016)
    stack_installation_factor = lookup_value(scenario, 'stack_installation_factor', year)
    baseline_installed_stack_capital_cost = stack_installation_factor * baseline_uninstalled_stack_capital_cost

    baseline_uninstalled_mechanical_BoP_cost = CEPCI_inflator * consumer_price_inflator * mechanical_BoP_unit_cost * peak_daily_production_rate  # ($2016)
    mechanical_BoP_installation_factor = 1
    baseline_installed_mechanical_BoP_cost = mechanical_BoP_installation_factor * baseline_uninstalled_mechanical_BoP_cost

    baseline_uninstalled_electrical_BoP_cost = CEPCI_inflator * consumer_price_inflator * electrical_BoP_cost * stack_input_power * 1000  # ($2016)
    electrical_BoP_installation_factor = lookup_value(scenario, 'electrical_BoP_installation_factor', year)
    baseline_installed_electrical_BoP_cost = electrical_BoP_installation_factor * baseline_uninstalled_electrical_BoP_cost

    baseline_total_installed_cost = baseline_installed_stack_capital_cost + baseline_installed_mechanical_BoP_cost + baseline_installed_electrical_BoP_cost

    # ------------------------------------------------PLANT SCALING-------------------------------------------------------------------#

    new_plant_output = avg_daily_H2_production/cap_factor # kg H2/day
    scale_ratio = new_plant_output/baseline_plant_output  # ratio of new design capacity to baseline design capacity (linear scaling)
    default_scaling_factor_exponent = 0.6  # discrepancy
    scale_factor = scale_ratio**default_scaling_factor_exponent  # rato of total scaled installed capital cost to total baseline installed capital cost (exponential scaling)
    lower_limit_for_scaling_capacity = 20000  # kgH2/day
    upper_limit_for_scaling_capacity = 200000  # kgH2/day

    scaled_uninstalled_stack_capital_cost = baseline_uninstalled_stack_capital_cost * scale_factor
    scaled_installed_stack_capital_cost = scaled_uninstalled_stack_capital_cost * stack_installation_factor

    scaled_uninstalled_mechanical_BoP_cost = baseline_uninstalled_mechanical_BoP_cost * scale_factor
    scaled_installed_mechanical_BoP_cost = scaled_uninstalled_mechanical_BoP_cost * mechanical_BoP_installation_factor

    scaled_uninstalled_electrical_BoP_cost = baseline_uninstalled_electrical_BoP_cost * scale_factor
    scaled_installed_electrical_BoP_cost = scaled_uninstalled_electrical_BoP_cost * electrical_BoP_installation_factor

    scaled_total_installed_cost = scaled_installed_stack_capital_cost + scaled_installed_mechanical_BoP_cost + scaled_installed_electrical_BoP_cost

    # -------------------------------------------------------H2A INPUT-------------------------------------------------------------#

    # --------------------------------------Capital Costs--------------------------------------------------------#

    H2A_total_direct_capital_cost = int(scaled_total_installed_cost)
    cost_scaling_factor = 1  # combined plant scaling and escalation factor

    # ------------Indirect Depreciable Capital Costs---------------------#
    site_preparation = 0.02 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor
    engineering_design = 0.1 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor
    project_contingency = 0.15 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor
    upfront_permitting_cost = 0.15 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor

    total_depreciable_costs = int(
        H2A_total_direct_capital_cost + site_preparation + engineering_design + project_contingency + upfront_permitting_cost)

    # ------------Non Depreciable Capital Costs---------------------#
    cost_of_land = 50000  # ($2016)/acre
    land_required = 5  # acres
    land_cost = cost_of_land * land_required
    other_nondepreciable_cost = 0

    total_nondepreciable_costs = land_cost + other_nondepreciable_cost

    total_capital_costs = total_depreciable_costs + total_nondepreciable_costs

    # --------------------------------------Fixed Operating Costs------------------------------------------------#
    total_plant_staff = 10  # number of FTEs employed by plant
    burdened_labor_cost = 50  # including overhead ($/man-hr)
    labor_cost = total_plant_staff * burdened_labor_cost * 2080  # ($2016)/year

    GA_rate = 20  # percent labor cos (general and admin)
    GA_cost = labor_cost * (GA_rate / 100)  # $/year
    licensing_permits_fees = 0  # $/year
    propertytax_insurancerate = 2  # percent of total capital investment per year
    propertytax_insurancecost = total_capital_costs * (propertytax_insurancerate / 100)  # $/year
    rent = 0  # $/year
    material_costs_for_maintenance = 0.03 * H2A_total_direct_capital_cost / (CEPCI_inflator * consumer_price_inflator)
    production_maintenance_and_repairs = 0  # $/year
    other_fees = 0  # $/year
    other_fixed_OM_costs = 0  # $/year

    total_fixed_operating_costs = int(labor_cost + GA_cost + licensing_permits_fees + propertytax_insurancecost + rent \
                                      + material_costs_for_maintenance + production_maintenance_and_repairs + other_fees + other_fixed_OM_costs)

    # --------------------------------------Variable Operating Costs----------------------------------------------#

    # ------------------Other Material and Byproduct---------------------#

    Material_1 = 'Processed Water'  # feed
    processed_water_cost = 0.002375  # ($2016)/gal
    water_usage_per_kgH2 = 3.78  # usageperkgH2
    
    
    return basis_year_for_capital_cost, total_capital_costs, total_fixed_operating_costs, water_usage_per_kgH2, total_system_electrical_usage