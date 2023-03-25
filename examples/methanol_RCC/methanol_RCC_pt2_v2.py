import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def calc_lcoe(OCC_kw, FOM_kwyr, VOM_mwh, TASC_multiplier, discount_rate):
    
    for arg in [OCC_kw, FOM_kwyr, VOM_mwh, TASC_multiplier, discount_rate]:
        arg = np.array(arg)
    
    TCC_kw = np.multiply(OCC_kw,TASC_multiplier)
    TCC_recovery_kwyr = np.multiply(TCC_kw,discount_rate)
    TFC_kwyr = np.add(TCC_recovery_kwyr,FOM_kwyr)
    TFC_kwh = np.divide(TFC_kwyr,8760)
    VOM_kwh = np.divide(VOM_mwh,1000)
    lcoe_kwh = np.add(TFC_kwh,VOM_kwh)

    return lcoe_kwh


## Load dicts from json dumpfiles

current_dir = Path(__file__).parent.absolute()
resource_dir = current_dir/'..'/'resource_files'/'methanol_RCC'
with open(Path(resource_dir/'engin.json'),'r') as file:
    engin = json.load(file)
with open(Path(resource_dir/'finance.json'),'r') as file:
    finance = json.load(file)
with open(Path(resource_dir/'scenario.json'),'r') as file:
    scenario_info = json.load(file)
    plant_scenarios = scenario_info['plant_scenarios']
    cambium_scenarios = scenario_info['cambium_scenarios']
    cambium_scenario = scenario_info['cambium_scenario']
    atb_scenarios = scenario_info['atb_scenarios']
    H2A_scenarios = scenario_info['H2A_scenarios']
    MeOH_scenarios = scenario_info['MeOH_scenarios']
with open(Path(resource_dir/'locations.json'),'r') as file:
    locations = json.load(file)
with open(Path(resource_dir/('cambium_prices_'+cambium_scenario+'.json')),'r') as file:
    cambium_prices = pd.read_json(file)
with open(Path(resource_dir/('cambium_ghgs_'+cambium_scenario+'.json')),'r') as file:
    cambium_ghgs = pd.read_json(file)
with open(Path(resource_dir/('cambium_ng_'+cambium_scenario+'.json')),'r') as file:
    cambium_ng = pd.read_json(file)
with open(Path(resource_dir/('cambium_aeo_multipliers.json')),'r') as file:
    state_multipliers = json.load(file)

## Downselect locations dict to one location

site_name = scenario_info['site_selection']['site_name']
site_num = scenario_info['site_selection']['site_num']
location = {}
for key, value in locations[site_name].items():
    if type(value) is not dict:
        location[key] = value[site_num-1]
    else:
        location[key] = {}
        for sub_key, sub_value in value.items():
            location[key][sub_key] = sub_value[site_num-1]


# ## Size renewable plants using location data

# pv_out_kw = location['pv_output_kw']
# pv_cap_kw = location['pv_capacity_kw']
# wind_out_kw = location['wind_output_kw']
# wind_cap_kw = location['wind_capacity_kw']
# if location['on_land']:
#     engin['PV']['output_kw'] = {plant_scenarios['PV']:pv_out_kw}
#     engin['PV']['capacity_kw'] = {plant_scenarios['PV']:pv_cap_kw}
#     engin['LBW']['output_kw'] = {plant_scenarios['LBW']:wind_out_kw}
#     engin['LBW']['capacity_kw'] = {plant_scenarios['LBW']:wind_cap_kw}
#     engin['OSW']['output_kw'] = {plant_scenarios['OSW']:0}
#     engin['OSW']['capacity_kw'] = {plant_scenarios['OSW']:0}
# else:
#     engin['PV']['output_kw'] = {plant_scenarios['PV']:0}
#     engin['PV']['capacity_kw'] = {plant_scenarios['PV']:0}
#     engin['LBW']['output_kw'] = {plant_scenarios['LBW']:0}
#     engin['LBW']['capacity_kw'] = {plant_scenarios['LBW']:0}
#     engin['OSW']['output_kw'] = {plant_scenarios['OSW']:wind_out_kw}
#     engin['OSW']['capacity_kw'] = {plant_scenarios['OSW']:wind_cap_kw}


## Set the H2 variable costs using electricity cost

for plant in ['HCO2','HPSR']:
    orig_lcoe_kwh = location[plant]['orig_lcoe_$_kwh']
    lcoe_kwh = location[plant]['lcoe_$_kwh']
    h2_scenario = plant_scenarios[plant]
    kw_in = engin[plant]['elec_in_kw'][h2_scenario]
    kwh_yr = [i*8760 for i in kw_in]
    mwh_yr = [i/1000 for i in kwh_yr]
    lcoe_yr = list(np.multiply(kwh_yr,lcoe_kwh))
    finance[plant]['VOM_elec_$_yr'] = {h2_scenario:lcoe_yr}
    h2_output_kw = engin[plant]['output_kw'][h2_scenario]
    mwh_yr = h2_output_kw*8.76
    VOM_elec_mwh = list(np.divide(lcoe_yr,mwh_yr))
    finance[plant]['VOM_elec_$_mwh'] = {h2_scenario:VOM_elec_mwh}

    sim_years = scenario_info['sim_years']
    VOM_comps = ['VOM_H2O_$_mwh',
                'VOM_elec_$_mwh']
    output_kw = engin[plant]['output_kw'][h2_scenario]
    mwh_yr = output_kw*8.76
    finance[plant]['VOM_$_mwh'] = {h2_scenario:[]}
    finance[plant]['VOM_$_yr'] = {h2_scenario:[]}
    for i, year in enumerate(sim_years):
        VOM_mwh = 0
        for VOM_comp in VOM_comps:
            if type(finance[plant][VOM_comp]) is dict:
                if type(finance[plant][VOM_comp][h2_scenario]) is list:
                    VOM_mwh += finance[plant][VOM_comp][h2_scenario][i]
                else:
                    VOM_mwh += finance[plant][VOM_comp][h2_scenario]
            else:
                VOM_mwh += finance[plant][VOM_comp]
        year_VOM = copy.copy(VOM_mwh)
        finance[plant]['VOM_$_mwh'][h2_scenario].append(year_VOM)
        finance[plant]['VOM_$_yr'][h2_scenario].append(year_VOM*mwh_yr)


## Get CO2 production cost from difference between NGCC and CCS plant LCOE

NG_plants = ['NGCC','CCS']

for NG_plant in NG_plants:
    NG_scenario = plant_scenarios[NG_plant]
    OCC_kw = finance[NG_plant]['OCC_$_kw'][NG_scenario]
    FOM_kwyr = finance[NG_plant]['FOM_$_kwyr']
    VOM_mwh = finance[NG_plant]['VOM_$_mwh'][NG_scenario]
    discount_rate = finance[NG_plant]['discount_rate']
    TASC_multiplier = finance[NG_plant]['TASC_multiplier']
    ng_lcoe_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,TASC_multiplier,discount_rate)
    finance[NG_plant]['lcoe_$_kwh'] = {NG_scenario:ng_lcoe_kwh}
lcoe_ngcc = list(finance['NGCC']['lcoe_$_kwh'][plant_scenarios['NGCC']])
lcoe_ccs = list(finance['CCS']['lcoe_$_kwh'][plant_scenarios['CCS']])
lcoe_increase_kwh = list(np.subtract(lcoe_ccs,lcoe_ngcc))
CO2_out_kg_mwh = engin['CCS']['CO2_out_kg_mwh']
CO2_price_kg = [i/CO2_out_kg_mwh*1000 for i in lcoe_increase_kwh]
finance['CCS']['CO2_$_kg'] = {plant_scenarios['CCS']:CO2_price_kg}
for plant in ['MSMR','MSMC','MCO2','MPSR']:
    MeOH_scenario = plant_scenarios[plant]
    output_kw = engin[plant]['output_kw'][MeOH_scenario]
    output_mwh_yr = output_kw*8.76
    if 'SM' in plant:
        elec_out_mwh_yr = engin[plant]['elec_out_mwh_yr'][MeOH_scenario]
        VOM_elec_yr = [-elec_out_mwh_yr*1000*i for i in lcoe_ngcc]
        VOM_elec_mwh = [i/output_mwh_yr for i in VOM_elec_yr]
        finance[plant]['VOM_elec_$_yr'] = {MeOH_scenario:VOM_elec_yr}
        finance[plant]['VOM_elec_$_mwh'] = {MeOH_scenario:VOM_elec_mwh}
    else:
        finance[plant]['CO2_cost_$_kg'] = {MeOH_scenario:CO2_price_kg}
        CO2_kg_yr_in = engin[plant]['CO2_kg_yr_in'][MeOH_scenario]
        VOM_CO2_yr = [CO2_kg_yr_in*i for i in CO2_price_kg]
        VOM_CO2_mwh = list(np.divide(VOM_CO2_yr,np.multiply(output_kw,8.67)))
        finance[plant]['VOM_CO2_$_yr'] = {MeOH_scenario:VOM_CO2_yr}
        finance[plant]['VOM_CO2_$_mwh'] = {MeOH_scenario:VOM_CO2_mwh}


## Get H2 production cost

for plant in ['HCO2','HPSR']:
    OCC_kw = finance[plant]['OCC_$_kw'][h2_scenario]
    FOM_kwyr = finance[plant]['FOM_$_kwyr'][h2_scenario]
    VOM_mwh = finance[plant]['VOM_$_mwh'][h2_scenario]
    discount_rate = finance[plant]['discount_rate']
    TASC_multiplier = finance[plant]['TASC_multiplier']
    lcoh_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,TASC_multiplier,discount_rate)
    finance[plant]['lcoh_$_kwh'] = {h2_scenario:lcoh_kwh}
    H2_LHV_MJ_kg = engin[plant]['H2_LHV_MJ_kg']
    lcoh_kg = list(np.multiply(lcoh_kwh,H2_LHV_MJ_kg/3600*1000))
    finance[plant]['lcoh_$_kg'] = lcoh_kg
    finance['M'+plant[1:]]['VOM_H2_$_kgH2'] = {MeOH_scenario:lcoh_kg}


## Calculate MeOH production cost

for plant in ['MSMR','MSMC','MCO2','MPSR']:
    sim_years = scenario_info['sim_years']
    MeOH_LHV_MJ_kg = engin[plant]['MeOH_LHV_MJ_kg']
    output_kw = engin[plant]['output_kw'][MeOH_scenario]
    mwh_yr = output_kw*8.76
    if 'SM' in plant:
        VOM_comps = ['VOM_elec_$_mwh',
                    'VOM_NG_$_mwh',
                    'VOM_H2O_$_mwh',
                    'VOM_cat_$_mwh',
                    'VOM_TS_$_mwh',
                    'VOM_other_$_mwh']
    else:
        cambium_aeo_multiplier = state_multipliers[site_name[:2]]
        cambium_price = list(cambium_prices.loc[site_name[:2],sim_years[0]:].values)
        cambium_price.extend([cambium_price[-1]]*(len(sim_years)-
                                                  len(cambium_prices.loc[site_name[:2]].values)))
        elec_price_kwh = np.mean([i*cambium_aeo_multiplier/1000 for i in cambium_price])
        elec_in_mwh_yr = engin[plant]['elec_in_mwh_yr'][MeOH_scenario]
        VOM_elec_yr = elec_price_kwh*1000*elec_in_mwh_yr
        VOM_elec_mwh = VOM_elec_yr/mwh_yr
        finance[plant]['VOM_elec_$_yr'] = {MeOH_scenario:VOM_elec_yr}
        finance[plant]['VOM_elec_$_mwh'] = {MeOH_scenario:VOM_elec_mwh}
        H2_kg_yr = engin[plant]['H2_kg_yr_in'][MeOH_scenario]
        H2_price_kg = finance[plant]['VOM_H2_$_kgH2'][MeOH_scenario]
        VOM_H2_yr = list(np.multiply(H2_price_kg,H2_kg_yr))
        finance[plant]['VOM_H2_$_yr'] = {MeOH_scenario:VOM_H2_yr}
        VOM_H2_mwh = [i/mwh_yr for i in VOM_H2_yr]
        finance[plant]['VOM_H2_$_mwh'] = {MeOH_scenario:VOM_H2_mwh}
        VOM_comps = ['VOM_elec_$_mwh',
                    'VOM_H2_$_mwh',
                    'VOM_CO2_$_mwh',
                    'VOM_H2O_$_mwh',
                    'VOM_cat_$_mwh',
                    'VOM_other_$_mwh']
    finance[plant]['VOM_$_mwh'] = {MeOH_scenario:[]}
    finance[plant]['VOM_$_yr'] = {MeOH_scenario:[]}
    for i, year in enumerate(sim_years):
        VOM_mwh = 0
        for VOM_comp in VOM_comps:
            if type(finance[plant][VOM_comp]) is dict:
                if type(finance[plant][VOM_comp][MeOH_scenario]) is list:
                    VOM_mwh += finance[plant][VOM_comp][MeOH_scenario][i]
                else:
                    VOM_mwh += finance[plant][VOM_comp][MeOH_scenario]
            else:
                VOM_mwh += finance[plant][VOM_comp]
        year_VOM = copy.copy(VOM_mwh)
        finance[plant]['VOM_$_mwh'][MeOH_scenario].append(year_VOM)
        finance[plant]['VOM_$_yr'][MeOH_scenario].append(year_VOM*mwh_yr)
    OCC_kw = finance[plant]['OCC_$_kw'][MeOH_scenario]
    FOM_kwyr = finance[plant]['FOM_$_kwyr'][MeOH_scenario]
    VOM_mwh = finance[plant]['VOM_$_mwh'][MeOH_scenario]
    discount_rate = finance[plant]['discount_rate']
    TASC_multiplier = finance[plant]['TASC_multiplier']
    lcom_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,TASC_multiplier,discount_rate)
    finance[plant]['lcom_$_kwh'] = {MeOH_scenario:lcom_kwh}
    lcom_kg = list(np.multiply(lcom_kwh,MeOH_LHV_MJ_kg/3600*1000))
    finance[plant]['lcom_$_kg'] = lcom_kg


# # Import TRACI impact analysis results for individual plants
# elec_rows = [0]
# elec_rows.extend(list(np.arange(28,37)))
# co2_rows = list(np.arange(0,28))
# co2_rows.extend(list(np.arange(31,37)))
# h2_rows = list(np.arange(0,31))
# h2_rows.extend(list(np.arange(34,37)))
# methanol_rows = list(np.arange(0,34))
# elec_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=elec_rows)
# elec_lca_df = elec_lca_df.iloc[:,:7]
# co2_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=co2_rows)
# co2_lca_df = co2_lca_df.iloc[:,:7]
# h2_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=h2_rows)
# h2_lca_df = h2_lca_df.iloc[:,:7]
# methanol_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=methanol_rows)
# methanol_lca_df = methanol_lca_df.iloc[:,:7]

# # Get impact factor units
# factor_names = ['Eutrophication Potential',
#                 'Acidification Potential',
#                 'Particulate Matter Formation Potential',
#                 'Photochemical Smog Formation Potential',
#                 'Global Warming Potential',
#                 'Ozone Depletion Potential',
#                 'Water Consumption']
# traci_factors = {}
# traci_units = elec_lca_df.columns.values
# for i, name in enumerate(factor_names):
#     traci_factors[name] = traci_units[i][:-4]

# # Interpolate grid LCA for site over 2020-2050 from 5 year intervals
# grid_lca = pd.DataFrame(np.zeros((31,7)),index=np.arange(2020,2051),columns=traci_units)
# five_yrs = np.arange(20,55,5)
# for five_yr in five_yrs:
#     df_index = site_name+str(five_yr)
#     grid_lca.loc[five_yr+2000] = elec_lca_df.loc[df_index]
#     if five_yr > 20:
#         for i, year in enumerate(np.arange(five_yr-4,five_yr)):
#             prev_lca = grid_lca.loc[five_yr-5+2000].values
#             next_lca = grid_lca.loc[five_yr+2000].values
#             year_lca = np.zeros(len(factor_names))
#             for j in range(len(factor_names)):
#                 year_lca[j] = prev_lca[j] + (next_lca[j]-prev_lca[j])*(i+1)/5
#             grid_lca.loc[year+2000] = year_lca

# # Put lca prices in dict
# lca = {}
# lca['Grid'] = {}
# elec_techs = ['NGCC','CCS','PV','LBW','OSW']
# for plant in elec_techs:
#     lca[plant] = {}
#     for unit in elec_lca_df.columns.values:
#         lca[plant][unit] = elec_lca_df.loc[plant,unit]
#         grid_lca_list = []
#         for year in sim_years:
#             grid_lca_list.append(grid_lca.loc[year,unit])
#         lca['Grid'][unit] = grid_lca_list
# for plant in ['NGCC','CCS']:
#     for unit in co2_lca_df.columns.values:
#         lca[plant][unit] = co2_lca_df.loc[plant,unit]
# for plant in ['H2St','H2NG']:
#     lca[plant] = {}
#     for unit in h2_lca_df.columns.values:
#         lca[plant][unit] = h2_lca_df.loc[plant,unit]
# for plant in ['MeRe','MeNG']:
#     lca[plant] = {}
#     for unit in methanol_lca_df.columns.values:
#         lca[plant][unit] = methanol_lca_df.loc['MeNG',unit]


# # Calculate impact factors per unit methanol
# lca['CO2'] = {}
# lca['H2'] = {}
# lca['MeOH'] = {}
# for unit in lca['MeNG'].keys():

#     MeOH_out = engin['MeOH']['MeOH_kg_yr'][MeOH_scenario]
    
#     # CO2 production - impact added by CCS plant over NGCC plant (CCS captured CO2)
#     # OR zero impact (NGCC flue gas) - NGCC flue gas impact attributed to ELECTRICITY out to grid
#     CO2_in = engin['MeOH']['CO2_kg_yr_in'][MeOH_scenario]
#     if 'CO2 Capture' in MeOH_scenario:
#         CCS_CO2_kgMeOH = lca['CCS'][unit[:-4]+'CO2']*CO2_in/MeOH_out
#         NGCC_CO2_kgMeOH = lca['NGCC'][unit[:-4]+'CO2']*CO2_in/MeOH_out
#         lca['CCS'][unit] = CCS_CO2_kgMeOH-NGCC_CO2_kgMeOH
#         lca['CO2'][unit] = CCS_CO2_kgMeOH-NGCC_CO2_kgMeOH
#     else:
#         lca['NGCC'][unit] = 0
#         lca['CO2'][unit] = 0
    
#     # H2 production - hybrid plant + grid electricity
#     H2_kg_in = engin['MeOH']['H2_kg_yr_in'][MeOH_scenario]
#     H2st_kgMeOH = lca['H2St'][unit[:-4]+'H2']*H2_kg_in/MeOH_out
#     lca['H2St'][unit] = H2st_kgMeOH
#     pv_kw_out = locations[site_name]['pv_output_kw'][site_num-1]
#     wind_kw_out = locations[site_name]['wind_output_kw'][site_num-1]
#     wind_plant = 'LBW' if location['on_land'] else 'OSW'
#     for plant in ['PV',wind_plant,'Grid','H2']:
#         lca[plant][unit[:-4]+'H2'] = []
#         lca[plant][unit] = []
#     for i, year in enumerate(sim_years):
#         pct_pv = 100*pv_kw_out[i]/(pv_kw_out[i]+wind_kw_out[i])
#         pct_wind = 100-pct_pv
#         H2_MWh_yr = location[plant]['electrolyzer_input_kw'][i]*8.76
#         pv_em_MWh = lca['PV'][unit[:-6]+'MWh']*pct_pv/100
#         wind_em_MWh = lca[wind_plant][unit[:-6]+'MWh']*pct_wind/100
#         hyb_em_MWh = pv_em_MWh+wind_em_MWh
#         grid_em_MWh = grid_lca.loc[year,unit[:-6]+'MWh']
#         buy_em_MWh = grid_em_MWh-hyb_em_MWh
#         sell_em_MWh = hyb_em_MWh-grid_em_MWh
#         grid_bought_MWh_yr = location[plant]['grid_bought_kw'][i]*8.76
#         grid_sold_MWh_yr = location[plant]['grid_sold_kw'][i]*8.76
#         buy_em_kg_yr = buy_em_MWh*grid_bought_MWh_yr
#         sell_em_kg_yr = sell_em_MWh*grid_sold_MWh_yr
#         elyzer_hyb_MWh_yr = H2_MWh_yr-grid_bought_MWh_yr
#         lca['PV'][unit[:-4]+'H2'].append(elyzer_hyb_MWh_yr*pct_pv/100*hyb_em_MWh/H2_kg_in)
#         lca[wind_plant][unit[:-4]+'H2'].append(elyzer_hyb_MWh_yr*pct_wind/100*hyb_em_MWh/H2_kg_in)
#         lca['Grid'][unit[:-4]+'H2'].append((buy_em_kg_yr+sell_em_kg_yr)/H2_kg_in)
#         lca['H2'][unit[:-4]+'H2'].append(lca['PV'][unit[:-4]+'H2'][i] + \
#                                          lca[wind_plant][unit[:-4]+'H2'][i] + \
#                                          lca['Grid'][unit[:-4]+'H2'][i] +
#                                          lca['H2St'][unit[:-4]+'H2'])
#         for plant in ['PV',wind_plant,'Grid','H2']:
#             lca[plant][unit].append(lca[plant][unit[:-4]+'H2'][i]*H2_kg_in/MeOH_out)

#     # Add together components of methanol lca: H2 in, CO2 in, and reactor
#     lca['MeOH'][unit] = []
#     for i, year in enumerate(sim_years):
#         lca['MeOH'][unit] = lca['CO2'][unit] + lca['H2'][unit] + lca['MeRe'][unit]


# Print prices per unit for ALL scenarios
prices_to_check = ['OCC_$_kw','FOM_$_kwyr','VOM_$_mwh']
plants_to_check = plant_scenarios.keys()
# for plant in plants_to_check:
#     if plant == 'H2':
#         scenario_list = H2A_scenarios
#     elif plant == 'MeOH':
#         scenario_list = MeOH_scenarios
#     else:
#         scenario_list = atb_scenarios
#     for scenario in scenario_list:
#         for price in prices_to_check:
#             try:
#                 price_value = finance[plant][price]
#                 if type(price_value) is dict:
#                     price_value = price_value[scenario]
#                     if type(price_value) is list:
#                         price_value = price_value[0]
#                 print('{} plant {} ({} scenario): ${}'.format(plant,
#                                                                 price,
#                                                                 scenario,
#                                                                 price_value))
#             except:
#                 print('{} plant did not import {} for {} scenario'.format(plant,
#                                                                         price,
#                                                                         scenario))

# # Print absolute prices and parameters for specific scenarios
# prices_to_check = ['OCC_$','FOM_$_yr','VOM_$_yr']
# params_to_check = ['output_kw'] 
# for plant in plants_to_check:
#     scenario = plant_scenarios[plant]
#     for price in prices_to_check:
#         try:
#             price_value = finance[plant][price]
#             if type(price_value) is dict:
#                 price_value = price_value[scenario]
#                 if type(price_value) is list:
#                     price_value = price_value[0]
#             print('{} plant {} ({} scenario): ${}'.format(plant,
#                                                             price,
#                                                             scenario,
#                                                             price_value))
#         except:
#             print('{} plant did not import {} for {} scenario'.format(plant,
#                                                                     price,
#                                                                     scenario))
#     for param in params_to_check:
#         try:
#             param_value = engin[plant][param]
#             if type(param_value) is dict:
#                 param_value = param_value[scenario]
#                 if type(param_value) is list:
#                     param_value = param_value[0]
#             print('{} plant {} ({} scenario): {}'.format(plant,
#                                                             param,
#                                                             scenario,
#                                                             param_value))
#         except:
#             print('{} plant did not import {} for {} scenario'.format(plant,
#                                                                     param,
#                                                                     scenario))


# Print absolute prices and parameters for specific scenarios
prices_to_check = ['OCC_$','FOM_$_yr','VOM_$_mwh','VOM_other_$_mwh','VOM_elec_$_mwh','VOM_H2_$_mwh','VOM_CO2_$_mwh','VOM_NG_$_mwh','VOM_H2O_$_mwh','VOM_cat_$_mwh']
params_to_check = ['output_kw','elec_out_mwh_yr','H2_kg_yr','CO2_kg_yr','MeOH_kg_yr','NG_kg_yr_in'] 
for plant in plants_to_check:
    scenario = plant_scenarios[plant]
    for price in prices_to_check:
        try:
            price_value = finance[plant][price]
            if type(price_value) is dict:
                price_value = price_value[scenario]
                if type(price_value) is list:
                    price_value = price_value[0]
            print('{} plant {} ({} scenario): ${:,.2f}'.format(plant,
                                                            price,
                                                            scenario,
                                                            price_value))
        except:
            print('{} plant did not import {} for {} scenario'.format(plant,
                                                                    price,
                                                                    scenario))
    for param in params_to_check:
        # if 'prod' in param:
        #     if 'CC' in plant:
        #         product = 'CO2'
        #     elif plant[0] == 'H':
        #         product = 'H2'
        #     elif plant[0] == 'M':
        #         product = 'MeOH'
        #     else:
        #         product = 'elec'
        #     if product == 'elec':
        #         param = 'elec_out_mwh_yr'
        #     else:
        #         param = param = product+'_kg_yr'
        try:
            param_value = engin[plant][param]
            if type(param_value) is dict:
                param_value = param_value[scenario]
                if type(param_value) is list:
                    param_value = param_value[0]
            print('{} plant {} ({} scenario): {:,.2f}'.format(plant,
                                                            param,
                                                            scenario,
                                                            param_value))
        except:
            print('{} plant did not import {} for {} scenario'.format(plant,
                                                                    param,
                                                                    scenario))

labels = ['NGCC w/o carbon capture',
        'NGCC with carbon capture',
        'Wind/solar w/o grid exchange',
        'Wind/solar with grid exchange']

plots = [lcoe_ngcc,lcoe_ccs,orig_lcoe_kwh,lcoe_kwh]

plt.subplot(2,2,1)
for i, label in enumerate(labels):
    plt.plot(sim_years,plots[i],label=label)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Levelized Cost [2020 $/kWh]')
plt.title('Electricity')

labels = ['CO2 captured from NGCC plant',
          'H2 produced from electrolyzer',
          'Methanol produced by Nyari et al. plant']

labels = ['Baseline #0: SMR without Carbon Capture',
          'Baseline #1: SMR with Carbon Capture',
          'Baseline #2: CO2 Hydrogenation State of Art',
          'Novel Technology: NREL PSRs']


mass_ratio_CO2_MeOH = 1.397 # kg CO2 in needed per kg of MeOH produced
mass_ratio_H2_MeOH = 0.199 # kg H2 in needed per kg of MeOH produced
lcoh_kg = [i*mass_ratio_H2_MeOH for i in lcoh_kg]
CO2_price_kg = [i*mass_ratio_CO2_MeOH for i in CO2_price_kg]

plots = [CO2_price_kg,lcoh_kg,lcom_kg]
plants = ['MSMR','MSMC','MCO2','MPSR']


plt.subplot(2,2,2)
for i, label in enumerate(labels):
    lcom = finance[plants[i]]['lcom_$_kg']
    plt.plot(sim_years,lcom,label=label)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Levelized Cost [2020 $/kg of Methanol]')
plt.title('Materials - PER UNIT METHANOL')
          

# plt.subplot(2,2,4)
# plt.plot([sim_years[0],sim_years[-1]],[lca['MeNG']['kgCO2e_kgMeOH'],lca['MeNG']['kgCO2e_kgMeOH']],
#          label='Methanol from Natural Gas')
# plt.plot(sim_years,lca['MeOH']['kgCO2e_kgMeOH'],label='Methanol from NGCC-captured CO2 + Green Hydrogen')
# plt.xlabel('Year')
# plt.ylabel('kg CO2-equivalent / kg methanol')
# plt.title('PROVISIONAL LCA')
# plt.legend()
# plt.gcf().set_tight_layout(True)
plt.show()