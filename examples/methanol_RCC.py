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
import numpy as np
from pathlib import Path
from global_land_mask import globe
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

sim_basis_year = 2020
sim_start_year = 2020
sim_end_year = 2050
sim_years = np.arange(sim_start_year,sim_end_year)

plant_lifespan = 30 # years
discount_rate = 0.07 # (= fixed charge rate)
TASC_multiplier = 1.093 # total overnight cost * TASC multiplier = total as-spent cost

NG_price_MMBTU = 4.56 # $/MMBTU TODO: make variable NG price scenarios
NG_LHV_MJ_kg = 47.1 # Natural gas net calorific value, MJ/kg
H2O_price_Tgal = 2.56 # $/Tgal TODO: make variable water price scenarios

select_locations = False # Switch to True to only analyze locations listed below
locations = {'IA': {'on_land':[True ,], 'lat':[43.094000,], 'lon':[-93.292220,]},
             'TX': {'on_land':[True ,], 'lat':[32.337679,], 'lon':[-97.734610,]},
             'NJ': {'on_land':[False ,], 'lat':[39.600000,], 'lon':[-73.400000,]},
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
land_rad = 60 # km radius of survey area around NGCC plant on land
osw_rad = 20 # km radium of survey area around offshore wind location

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
basis_year = 2020
pv_constants = {'array_type':2,
                'azimuth':180,
                'inv_eff':None,
                'dc_ac_ratio':1.28} # ILR NOT OPTIMIZED - can HOPP do this?
bos_ratio = {'LBW': 322/1462, 
             'PV': 20/89}
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
                        value = convert_dollar_year(value, basis_year, sim_basis_year)
                    values.append(value)
                param[scenario] = values

#endregion

## Import the standard NETL NGCC power plant <NETL-PUB-22638/B31A & B31B>
#region

# Values in NETL report
fullcap_MW = {'NGCC': 727, 'CCS': 646}
fullcap_C_kg_hr = {'NGCC': 67890, 'CCS': 61090}
fullcap_H2O_Tgal_day = {'NGCC': 2090, 'CCS': 3437}
CO2_kg_MWh = {}
H2O_Tgal_MWh = {}

# Scale NGCC plant with CCS
NGCC_out = {'NGCC': NGCC_out,
            'CCS': NGCC_out*fullcap_MW['CCS']/fullcap_MW['NGCC']}

# Convert to values per MWh produced
for key, value in fullcap_C_kg_hr.items():
    CO2_kg_MWh[key] = value*(C_MW+O_MW*2)/C_MW/fullcap_MW[key]
for key, value in fullcap_H2O_Tgal_day.items():
    H2O_Tgal_MWh[key] = value/24/fullcap_MW[key]

#endregion

## Import the NGCC plant locations and create list of survey locations
#region

earth_rad = 6371 # Earth's radium in km, needed for lat/long calcs

# Loop through locations (begin as list of 1 NGCC plant location)
for id, loc in locations.items():
    on_land = loc['on_land'][0]
    survey_rad = land_rad if on_land else osw_rad
    lat = loc['lat'][0]
    lon = loc['lon'][0]
    
    # Add inner circle of 6 surrounding locations @ half of survey radius
    in_circle_lat_delta = [3**.5/4, 0, -(3**.5/4), -(3**.5/4), 0,  3**.5/4]
    for i, lat_delta in enumerate(in_circle_lat_delta):
        lat_delta = survey_rad/earth_rad*lat_delta
        lon_delta = ((survey_rad**2/4/earth_rad**2-lat_delta**2)/\
                        (np.cos(lat/180*np.pi)**2))**.5*180/np.pi
        lat_delta *= 180/np.pi
        if i<3: lon_delta = -lon_delta 
        loc['lat'].append(lat+lat_delta)
        loc['lon'].append(lon+lon_delta)
        loc['on_land'].append(globe.is_land(lat+lat_delta,lon+lon_delta))

    # Add outer circle of 12 surrounding location @ full survey radius
    out_circle_lat_delta = [ 1,   3**.5/2,   .5, 0, -.5, -(3**.5/2),
                            -1, -(3**.5/2), -.5, 0,  .5,   3**.5/2]
    for i, lat_delta in enumerate(out_circle_lat_delta):
        lat_delta = survey_rad/earth_rad*lat_delta
        lon_delta = ((survey_rad**2/earth_rad**2-lat_delta**2)/\
                        (np.cos(lat/180*np.pi)**2))**.5*180/np.pi
        lat_delta *= 180/np.pi
        if i<6: lon_delta = -lon_delta 
        loc['lat'].append(lat+lat_delta)
        loc['lon'].append(lon+lon_delta)
        loc['on_land'].append(globe.is_land(lat+lat_delta,lon+lon_delta))

#endregion

## Plot survey locations to check
#region

# Stolen from gis.stackexchange.com/questions/156035
def merc_x(lon):
  r_major=6378137.000
  return r_major*(lon*np.pi/180)
def merc_y(lat):
  if lat>89.5:lat=89.5
  if lat<-89.5:lat=-89.5
  r_major=6378137.000
  r_minor=6356752.3142
  temp=r_minor/r_major
  eccent=(1-temp**2)**.5
  phi=(lat*np.pi/180)
  sinphi=np.sin(phi)
  con=eccent*sinphi
  com=eccent/2
  con=((1.0-con)/(1.0+con))**com
  ts=np.tan((np.pi/2-phi)/2)/con
  y=0-r_major*np.log(ts)
  return y

# Set up background image
plt.clf
bg_img = 'RCC search area new.png'
img = plt.imread(resource_dir/bg_img)
ax = plt.gca()
min_x = merc_x(-100)
max_x = merc_x(-67)
min_y = merc_y(25)
max_y = merc_y(46)
ax.imshow(img, extent=[min_x, max_x, min_y, max_y])
# Plot survey locations
for id, loc in locations.items():
    for n in range(len(loc['lat'])):
        lat = loc['lat'][n]
        lon = loc['lon'][n]
        color = [1*loc['on_land'][n],0,0]
        x = merc_x(lon)
        y = merc_y(lat)
        ax.plot(x,y,'.',color=color)
plt.xlim([min_x,max_x])
plt.ylim([min_y,max_y])
plt.show()

#endregion

## TODO: Import NGCC plant stream table from NETL report

## TODO: Import Wind/Solar/NGCC plant LCI tables from OpenLCA

## TODO: Import natural gas price scenario

## Fit curve to MeOH plant CAPEX data from IRENA report
#region

# IRENA report data <ISBN 978-92-9260-320-5>
capacity_kt_y =  [7,    90,    30,   440,  16.3, 50,  1800,  100]
capex_mil_kt_y = [3.19, 2.325, 1.15, 1.26, 0.98, 1.9, 0.235, 0.62]
basis_year = 2020
capex_mil_ky_y = [convert_dollar_year(i, basis_year, sim_basis_year) for i in capex_mil_kt_y]
sources = ['Hank 2018', 'Mignard 2003', 'Clausen 2010', 'Perez-Fortes 2016',
            'Rivera-Tonoco 2016',' Belloti 2019', 'Nyari 2020', 'Szima 2018']

# Fit curve
def exp_curve(x, a, b): return a*x**(-b)
coeffs, _ = curve_fit(exp_curve, capacity_kt_y, capex_mil_kt_y)
a_cap2capex_mil_kt_y = coeffs[0]
b_cap2capex_mil_kt_y = coeffs[1]

# # Plot to check
# log_x = np.linspace(np.log(np.min(capacity_kt_y)),np.log(np.max(capacity_kt_y)),100)
# plt.clf()
# plt.plot(np.power(np.e,log_x),
#         a_cap2capex_mil_kt_y*np.power(np.power(np.e,log_x),-b_cap2capex_mil_kt_y))
# plt.plot(capacity_kt_y,capex_mil_kt_y,'.')
# plt.xscale('log')
# plt.show()

#endregion

## Import MeOH ASPEN process model results to determine H2 requirements
# TODO: Replace static Nyari placeholder (doi:10.1016/j.jcou.2020.101166)
#           with executable Tan model fed by NGCC plant stack stream
# TODO: Euros-dollars function
#region

# Nyari FOM model - FCI = "Fixed capital investment"
fci_capex_ratio = 1/1.47
direct_labor_euro_person_year = 60000 
direct_indirect_ratio = 0.3
workers_kt_y = 56/1800
o_and_m_fci_ratio = 0.015
insurance_fci_ratio = 0.005
tax_fci_ratio = 0.005

# Nyari reactor performance
mass_ratio_CO2_MeOH = 1.397 # kg CO2 in needed per kg of MeOH produced
mass_ratio_H2_MeOH = 0.199 # kg H2 in needed per kg of MeOH produced
conv_eff_CO2 = 0.9837 # % of CO2 in converted to MeOH (rest comes out stack)
conv_eff_H2 = 1 # % of H2 in converted to MeOH
MeOH_purity = 0.999 # purity of MeOH produced (rest is assumed H2O)
cool_use_kWh_kg_MeOH = 0.81 # Cooling utility usage, kWh/kg MeOH produced
elec_use_kWh_kg_MeOH = 0.175 # Electricity usage, kWh/kg MeOH produced
CO2_emission = 0.023 # kg CO2 emitted directly per kg MeOH

#endregion

## Import H2A model and assumptions from current and future scenarios
#region

# Set spreadsheet location info
filenames = {'Current':'current-central-pem-electrolysis-2019-v3-2018.xlsm',
            'Future':'future-central-pem-electrolysis-2019-v3-2018.xlsm'}
sheet_name = 'Input_Sheet_Template'
cell_loc = {'cap_factor':   'C25',
            'cap_kg_day':   'C26',
            'startup_year': 'C31',
            'basis_year':  'C32',
            'elec_kWh_kgH2':'C68',
            'total_capex_$':'C106',
            'FOM_$_yr':     'E122',
            'H2O_gal_kgH2': 'D135'}
# Import values            
H2A_values = {'Current':{},'Future':{}}
for key, value_dict in H2A_values.items():
    filename = filenames[key]
    workbook = load_workbook(resource_dir/filename, read_only=True, data_only=True)
    worksheet = workbook[sheet_name]
    for key, cell in cell_loc.items():
        cell_value = worksheet[cell].value
        # Correct for inflation
        if key is 'basis_year':
            basis_year = cell_value
        if '$' in key:
            cell_value = convert_dollar_year(cell_value, basis_year, sim_basis_year)
        value_dict[key] = cell_value

# Interpolate values for simulation years
H2A_values_new = {'Current':{},'Future':{}} # Will contain lists of all simulation years
for key in H2A_values['Current'].keys():
    H2A_values_new['Current'][key] = []
    H2A_values_new['Future'][key] = []
current_year = H2A_values['Current']['startup_year']
future_year = H2A_values['Future']['startup_year']
gap = current_year
for year in sim_years:
    interp_frac = (year-current_year)/gap
    interp_frac = min([interp_frac,1])
    for key in H2A_values['Current'].keys():
        current_value = H2A_values['Current'][key]
        future_value = H2A_values['Future'][key]
        new_value = interp_frac*(future_value-current_value)+current_value
        H2A_values_new['Current'][key].append(current_value)
        H2A_values_new['Future'][key].append(new_value)
H2A_values = H2A_values_new

#endregion

## Scale plant sizes accordingly
#region

# Scale MeOH plant
MeOH_kt_yr = {'NGCC':0, 'CCS':0}
for ng_type in MeOH_kt_yr.keys():
    CO2_kt_yr = CO2_kg_MWh[ng_type]*NGCC_out[ng_type]*NGCC_cap*8760/1e6
    MeOH_kt_yr[ng_type] = CO2_kt_yr*mass_ratio_CO2_MeOH

# Scale H2 plant
H2_kt_yr = {'NGCC':0, 'CCS':0}
for ng_type in H2_kt_yr.keys():
    H2_kt_yr[ng_type] = MeOH_kt_yr[ng_type]*mass_ratio_H2_MeOH
H2_plant = {'NGCC':{'Current':{},'Future':{}},
            'CCS': {'Current':{},'Future':{}}}
for ng_type, scenario_dict in H2_plant.items():
    for scenario, value_dict in scenario_dict.items():
        to_be_continued = True
 
# Scale hybrid plant
       

#endregion