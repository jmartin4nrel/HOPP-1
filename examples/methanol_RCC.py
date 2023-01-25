'''
methanol_RCC.py

Generates a TEA for flue gas CO2 capture from an NGCC power plant and conversion
to methanol using hydrogen generated from a hybrid wind/solar plant nearby.
Uses NREL Annual Technology Baseline (ATB) future tech development scenarios.
Tracks all of the material streams and exports in a convenient format for LCA.

Author: Jonathan Martin <jonathan.martin@nrel.gov>
Uses ASPEN process model developed by Eric Tan <eric.tan@nrel.gov>

'''

## Import libraries needed
#region

import json
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string

from hybrid.sites import SiteInfo
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

#endregion

## Convert dollars to correct year
#region

def convert_dollar_year(dollars, original_year, new_year):

    # TODO: Adjust dollars using CPI

    return dollars
    
#endregion    

## ALL CHANGES HERE - SELECT SCENARIO COMBOS, MANUALLY OVERRIDE PRICING ASSUMPTIONS
#region

sim_dollar_year = 2020
sim_start_year = 2020
sim_end_year = 2050

plant_lifespan = 30 # years
discount_rate = 0.07 # (= fixed charge rate)
TASC_multiplier = 1.093 # total overnight cost * TASC multiplier = total as-spent cost

NG_price_MMBTU = 4.56 # $/MMBTU TODO: make variable NG price scenarios
NG_LHV_MJ_kg = 47.1 # Natural gas net calorific value, MJ/kg
H2O_price_Tgal = 2.56 # $/Tgal TODO: make variable water price scenarios

select_locations = False # Switch to True to only analyze locations listed below
#             ID    On/offshore  Lat   Long
locations = {'IA': [False, 43.094000, -93.292220],
             'TX': [False, 32.337679, -97.734610],
             'NJ': [True,  40.837500, -74.024400],
             }

manual_LCOE = False # Switch to True to override HOPP-calculated LCOE
LCOE_kWh = 0.04 # $/kWh
manual_LCOH = False # Switch to True to override HOPP-calculated LCOE
LCOH_kg = 0.02 # $/kg

# Switch dict definitions to use certain scenarios for different technologies
# Options for non-H2: 'Conservative', 'Moderate', 'Advanced' (based on ATB)
# Options for H2: 'Current', 'Future': 'Current' holds 2015 baseline H2A scenario,
#       'Future' interpolates between that and 2040 'Future' H2A scenario
scenarios = {'NGCC':'Conservative',
             'CCS': 'Conservative',
             'PV':  'Advanced',
             'Wind':'Advanced',
             'OSW': 'Advanced',
             'H2':  'Future'}

min_plant_dist = 120 # km, minimum distance between NGCC plants in survey
survey_rad = 120 # km radius of survey area around NGCC plant

NGCC_out = 100 # MW, output of NGCC plant to scale results to
NGCC_cap = 0.85 # capacity factor of NGCC plant to scale results to

resource_dir = Path(__file__).parent.absolute()/'resource_files'/'methanol_RCC'

#endregion

## Physical constants
#region

C_MW = 12.011 # g/mol C
H_MW = 1.008 # g/mol H
O_MW = 15.999 # g/mol O
L_gal = 3.78541 # L/gal
kg_lb = 0.453592 # kg/lb
kJ_BTU = 1.05506 # kJ/BTU

#endregion

## Import the NREL ATB scenario pricing info
#region

# Set constants not given in spreadsheet
original_dollar_year = 2020
pv_constants = {'array_type':2,
                'azimuth':180,
                'inv_eff':None,
                'dc_ac_ratio':1.28} # ILR NOT OPTIMIZED - can HOPP do this?
wind_constants =   {'wind_turbine_max_cp':0.55,
                    'avail_bop_loss':0,
                    'avail_grid_loss':0,
                    'avail_turb_loss':0,
                    'elec_eff_loss':0,
                    'elec_parasitic_loss':0,
                    'env_degrad_loss':0,
                    'env_env_loss':0,
                    'env_icing_loss':0,
                    'ops_env_loss':0,
                    'ops_grid_loss':0,
                    'ops_load_loss':0,
                    'turb_generic_loss':0,
                    'turb_hysteresis_loss':0,
                    'turb_perf_loss':0,
                    'turb_specific_loss':0,
                    'wake_ext_loss':0}
osw_constants =   {'wind_turbine_max_cp':0.55,
                    'avail_bop_loss':0,
                    'avail_grid_loss':0,
                    'avail_turb_loss':0,
                    'elec_eff_loss':0,
                    'elec_parasitic_loss':0,
                    'env_degrad_loss':0,
                    'env_env_loss':0,
                    'env_icing_loss':0,
                    'ops_env_loss':0,
                    'ops_grid_loss':0,
                    'ops_load_loss':0,
                    'turb_generic_loss':0,
                    'turb_hysteresis_loss':0,
                    'turb_perf_loss':0,
                    'turb_specific_loss':0,
                    'wake_ext_loss':0}

# Set location info for Excel workbook with ATB values
filename = "2022 v3 Annual Technology Baseline Workbook Corrected 1-24-2023.xlsx"
sheets = {'NGCC':'Natural Gas_FE',
          'CCS': 'Natural Gas_FE',
          'PV':  'Solar - Utility PV',
          'LBW': 'Land-Based Wind',
          'OSW': 'Offshore Wind'}
base_cols =    {'NGCC':'M',
                'CCS': 'M',
                'PV':  'M',
                'LBW': 'M',
                'OSW': 'M'}
year_rows =    {'NGCC':68,
                'CCS': 68,
                'PV':  74,
                'LBW': 75,
                'OSW': 76}
# Rows with engineering/financial inputs (MAY NEED TO CHANGE IF HOPP/SAM CHANGES)
engin_atb ={'NGCC':{'heat_rate_BTU_Wh':{'Advanced':72,
                                        'Moderate':73,
                                        'Conservative':74}},
            'CCS': {'heat_rate_BTU_Wh':{'Advanced':75,
                                        'Moderate':76,
                                        'Conservative':77}},
            'PV':  {'bifaciality':     {'Advanced':75,
                                        'Moderate':76,
                                        'Conservative':77},
                    'albedo':          {'Advanced':78,
                                        'Moderate':79,
                                        'Conservative':80},
                    'losses':          {'Advanced':81,
                                        'Moderate':82,
                                        'Conservative':83},
                    'dc_degradation':  {'Advanced':84,
                                        'Moderate':85,
                                        'Conservative':86}},
            'LBW': {'hub_height':      {'Advanced':76,
                                        'Moderate':77,
                                        'Conservative':78},
                    'rotor_diameter':  {'Advanced':79,
                                        'Moderate':80,
                                        'Conservative':81},
                    'turbine_rating_kw':{'Advanced':82,
                                        'Moderate':83,
                                        'Conservative':84}},
            'OSW': {'hub_height':      {'Advanced':76,
                                        'Moderate':77,
                                        'Conservative':78},
                    'rotor_diameter':  {'Advanced':79,
                                        'Moderate':80,
                                        'Conservative':81},
                    'turbine_rating_kw':{'Advanced':82,
                                        'Moderate':83,
                                        'Conservative':84}}}
# OCC: Overnight capital cost ($/kW), FOM: Fixed O&M ($/kW-yr), VOM: Variable O&M ($/MWh)
finance_atb =  {'NGCC':{'OCC_$_kW':    {'Advanced':105,
                                        'Moderate':106,
                                        'Conservative':107},
                        'FOM_$_kWyr':  {'Advanced':116,
                                        'Moderate':117,
                                        'Conservative':118},
                        'VOM_$_MWh':   {'Advanced':127,
                                        'Moderate':128,
                                        'Conservative':129}},
                'CCS': {'OCC_$_kW':    {'Advanced':108,
                                        'Moderate':109,
                                        'Conservative':110},
                        'FOM_$_kWyr':  {'Advanced':119,
                                        'Moderate':120,
                                        'Conservative':121},
                        'VOM_$_MWh':   {'Advanced':130,
                                        'Moderate':131,
                                        'Conservative':132}},
                'PV':  {'OCC_$_kW':    {'Advanced':203,
                                        'Moderate':204,
                                        'Conservative':205},
                        'FOM_$_kWyr':  {'Advanced':235,
                                        'Moderate':236,
                                        'Conservative':237},
                        'VOM_$_MWh':   {'Advanced':267,
                                        'Moderate':268,
                                        'Conservative':269}},
                'LBW': {'OCC_$_kW':    {'Advanced':204,
                                        'Moderate':205,
                                        'Conservative':206},
                        'FOM_$_kWyr':  {'Advanced':236,
                                        'Moderate':237,
                                        'Conservative':238},
                        'VOM_$_MWh':   {'Advanced':268,
                                        'Moderate':269,
                                        'Conservative':270}},
                'OSW': {'OCC_$_kW':    {'Advanced':253,
                                        'Moderate':254,
                                        'Conservative':255},
                        'FOM_$_kWyr':  {'Advanced':297,
                                        'Moderate':298,
                                        'Conservative':299},
                        'VOM_$_MWh':   {'Advanced':341,
                                        'Moderate':342,
                                        'Conservative':343}}}

# Import from spreadsheet
workbook = load_workbook(resource_dir/filename, read_only=True, data_only=True)
atb_tech = ['NGCC','CCS','PV','LBW','OSW']
for tech in atb_tech:
    sheet_name = sheets[tech]
    worksheet = workbook[sheet_name]
    # Figure out column range for years
    base_col = base_cols[tech]
    base_col_idx = column_index_from_string(base_col)
    year_row = str(year_rows[tech])
    base_year = worksheet[base_col+year_row].value
    start_col_idx = base_col_idx+sim_start_year-base_year
    end_col_idx = base_col_idx+sim_end_year-base_year
    start_col = get_column_letter(start_col_idx)
    end_col = get_column_letter(end_col_idx)
    # Replace row numbers in nested dicts with engin/finance params
    for param_dict in [engin_atb, finance_atb]:
        for param in param_dict[tech].values():
            for scenario, row in param.items():
                cells = worksheet[start_col+str(row)+':'+end_col+str(row)][0]
                values = []
                for i, cell in enumerate(cells):
                    value = cell.value
                    if param_dict is finance_atb:
                        value = convert_dollar_year(value, original_dollar_year,
                                                            sim_dollar_year)
                    values.append(value)
                param[scenario] = values

#endregion

## Import the standard NETL NGCC power plant <NETL-PUB-22638/B31A & B31B>
#region

# Values in NETL report
fullcap_MW = {'NGCC': 727, 'CCS': 646}
fullcap_C_kg_hr = {'NGCC': 67890, 'CCS': 61090}
fullcap_H2O_Tgal_day = {'NGCC': 2090, 'CCS': 3437}
C_kg_MWh = {}
H2O_Tgal_MWh = {}

# Convert to values per MWh produced
for key, value in fullcap_C_kg_hr.items():
    C_kg_MWh[key] = value/fullcap_MW[key]
for key, value in fullcap_H2O_Tgal_day.items():
    H2O_Tgal_MWh[key] = value/24/fullcap_MW[key]

#endregion

## Import natural gas price scenario
#TODO

## Import the NGCC plant locations and create list of survey locations
#region

# Add inner circle of 6 surrounding locations
in_radius = survey_rad*.5

out_circle_x_mult = [0,  .5,   3**.5/2,   1,   3**.5/2,   .5, 0, 
                        -.5, -(3**.5/2), -1, -(3**.5/2), -.5, 0]

#endregion

## Import ASPEN process model results to determine H2 requirements


## Import H2A model and assumptions from current and future scenarios


## Scale plant sizes accordingly


