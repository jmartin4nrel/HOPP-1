'''
methanol_RCC.py

Generates a TEA for flue gas CO2 capture from an NGCC power plant and conversion
to methanol using hydrogen generated from a hybrid wind/solar plant nearby.
Uses NREL Annual Technology Baseline (ATB) future tech development scenarios.
Tracks all of the material streams and exports in a convenient format for LCA.

Author: Jonathan Martin <jonathan.martin@nrel.gov>
Uses ASPEN process model developed by Eric Tan <eric.tan@nrel.gov>

'''

## ALL CHANGES HERE - SELECT SCENARIO COMBOS, MANUALLY OVERRIDE PRICING ASSUMPTIONS

dollar_year = 2020
sim_start_year = 2025
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
             'NJ': [True,  40.837500, -74.024400]}

manual_LCOE = False # Switch to True to override HOPP-calculated LCOE
LCOE_kWh = 0.04 # $/kWh
manual_LCOH = False # Switch to True to override HOPP-calculated LCOE
LCOH_kg = 0.02 # $/kg

# Switch dict definitions to use certain scenarios for different technologies
# Options for non-H2: 'Conservative', 'Moderate', 'Advanced' (based on ATB)
# Options for H2: 'Current', 'Future': 'Current' holds 2015 baseline H2A scenario,
#       'Future' interpolates between that and 2040 'Future' H2A scenario
scenarios = {'NG':  'Conservative',
             'CC':  'Conservative',
             'Wind':'Advanced',
             'PV':  'Advanced',
             'H2':  'Future'}

min_plant_dist = 120 # km, minimum distance between NGCC plants in survey
survey_rad = 100 # km radius of survey area around NGCC plant

NGCC_out = 100 # MW, output of NGCC plant to scale results to


## Import libraries needed
#region

import json
from pathlib import Path

from hybrid.sites import SiteInfo
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

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


## Import the standard NETL NGCC power plant <NETL-PUB-22638/B31A & B31B>


## Import natural gas price scenario


## Import the NGCC plant locations and create list of survey locations


## Import ASPEN process model results to determine H2 requirements


## Import H2A model and assumptions from current and future scenarios


## Scale plant sizes accordingly