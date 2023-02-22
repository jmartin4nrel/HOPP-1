import json
import copy
import numpy as np
from pathlib import Path

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
    atb_scenarios = scenario_info['atb_scenarios']
    H2A_scenarios = scenario_info['H2A_scenarios']
    MeOH_scenarios = scenario_info['MeOH_scenarios']
with open(Path(resource_dir/'locations.json'),'r') as file:
    locations = json.load(file)


## Downselect locations dict to one location

site_name = scenario_info['site_selection']['site_name']
site_num = scenario_info['site_selection']['site_num']
location = {}
for key, value in locations[site_name].items():
    location[key] = value[site_num-1]


## Size renewable plants using location data

pv_out_kw = location['pv_output_kw']
pv_cap_kw = location['pv_capacity_kw']
wind_out_kw = location['wind_output_kw']
wind_cap_kw = location['wind_capacity_kw']
if location['on_land']:
    engin['PV']['output_kw'] = {plant_scenarios['PV']:pv_out_kw}
    engin['PV']['capacity_kw'] = {plant_scenarios['PV']:pv_cap_kw}
    engin['LBW']['output_kw'] = {plant_scenarios['LBW']:wind_out_kw}
    engin['LBW']['capacity_kw'] = {plant_scenarios['LBW']:wind_cap_kw}
    engin['OSW']['output_kw'] = {plant_scenarios['OSW']:0}
    engin['OSW']['capacity_kw'] = {plant_scenarios['OSW']:0}
else:
    engin['PV']['output_kw'] = {plant_scenarios['PV']:0}
    engin['PV']['capacity_kw'] = {plant_scenarios['PV']:0}
    engin['LBW']['output_kw'] = {plant_scenarios['LBW']:0}
    engin['LBW']['capacity_kw'] = {plant_scenarios['LBW']:0}
    engin['OSW']['output_kw'] = {plant_scenarios['OSW']:wind_out_kw}
    engin['OSW']['capacity_kw'] = {plant_scenarios['OSW']:wind_cap_kw}


## Set the H2 variable costs using electricity cost

lcoe_kwh = location['lcoe_$_kwh']
h2_scenario = plant_scenarios['H2']
kw_in = engin['H2']['elec_in_kw'][h2_scenario]
kwh_yr = [i*8760 for i in kw_in]
mwh_yr = [i/1000 for i in kwh_yr]
lcoe_yr = list(np.multiply(kwh_yr,lcoe_kwh))
finance['H2']['VOM_elec_$_yr'] = {h2_scenario:lcoe_yr}
h2_output_kw = engin['H2']['output_kw'][h2_scenario]
mwh_yr = h2_output_kw*8.76
VOM_elec_mwh = list(np.divide(lcoe_yr,mwh_yr))
finance['H2']['VOM_elec_$_mwh'] = {h2_scenario:VOM_elec_mwh}

sim_years = scenario_info['sim_years']
VOM_comps = ['VOM_H2O_$_mwh',
            'VOM_elec_$_mwh']
plant = 'H2'
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
    lcoe_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,TASC_multiplier,discount_rate)
    finance[NG_plant]['lcoe_$_kwh'] = {NG_scenario:lcoe_kwh}
MeOH_scenario = plant_scenarios['MeOH']
if 'Flue gas' in MeOH_scenario:
    CO2_price_kg = 0
else:
    lcoe_ngcc = finance['NGCC']['lcoe_$_kwh'][plant_scenarios['NGCC']]
    lcoe_ccs = finance['CCS']['lcoe_$_kwh'][plant_scenarios['CCS']]
    lcoe_increase_kwh = list(np.subtract(lcoe_ccs,lcoe_ngcc))
    CO2_out_kg_mwh = engin['CCS']['CO2_out_kg_mwh']
    CO2_price_kg = [i/CO2_out_kg_mwh*1000 for i in lcoe_increase_kwh]
    finance['CCS']['CO2_$_kg'] = {plant_scenarios['CCS']:CO2_price_kg}
finance['MeOH']['CO2_cost_$_kg'] = {MeOH_scenario:CO2_price_kg}
CO2_kg_yr_in = engin['MeOH']['CO2_kg_yr_in'][MeOH_scenario]
output_kw = engin['MeOH']['output_kw'][MeOH_scenario]
VOM_CO2_yr = [CO2_kg_yr_in*i for i in CO2_price_kg]
VOM_CO2_mwh = list(np.divide(VOM_CO2_yr,np.multiply(output_kw,8.67)))
finance['MeOH']['VOM_CO2_$_yr'] = {MeOH_scenario:VOM_CO2_yr}
finance['MeOH']['VOM_CO2_$_mwh'] = {MeOH_scenario:VOM_CO2_mwh}


## Get H2 production cost

OCC_kw = finance['H2']['OCC_$_kw'][h2_scenario]
FOM_kwyr = finance['H2']['FOM_$_kwyr'][h2_scenario]
VOM_mwh = finance['H2']['VOM_$_mwh'][h2_scenario]
discount_rate = finance['H2']['discount_rate']
TASC_multiplier = finance['H2']['TASC_multiplier']
lcoh_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,TASC_multiplier,discount_rate)
finance['H2']['lcoh_$_kwh'] = {h2_scenario:lcoh_kwh}
H2_LHV_MJ_kg = engin['H2']['H2_LHV_MJ_kg']
lcoh_kg = list(np.multiply(lcoh_kwh,H2_LHV_MJ_kg/3600*1000))
finance['H2']['lcoh_$_kg'] = lcoh_kg
finance['MeOH']['VOM_H2_$_kgH2'] = {MeOH_scenario:lcoh_kg}


## Calculate MeOH production cost

H2_kg_yr = engin['MeOH']['H2_kg_yr_in'][MeOH_scenario]
MeOH_LHV_MJ_kg = engin['MeOH']['MeOH_LHV_MJ_kg']
H2_price_kg = finance['MeOH']['VOM_H2_$_kgH2'][MeOH_scenario]
VOM_H2_yr = list(np.multiply(H2_price_kg,H2_kg_yr))
finance['MeOH']['VOM_H2_$_yr'] = {MeOH_scenario:VOM_H2_yr}
output_kw = engin['MeOH']['output_kw'][MeOH_scenario]
mwh_yr = output_kw*8.76
VOM_H2_mwh = [i/mwh_yr for i in VOM_H2_yr]
finance['MeOH']['VOM_H2_$_mwh'] = {MeOH_scenario:VOM_H2_mwh}
sim_years = scenario_info['sim_years']
VOM_comps = ['VOM_CO2_$_mwh',
            'VOM_H2_$_mwh']
plant = 'MeOH'
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
OCC_kw = finance['MeOH']['OCC_$_kw'][MeOH_scenario]
FOM_kwyr = finance['MeOH']['FOM_$_kwyr'][MeOH_scenario]
VOM_mwh = finance['MeOH']['VOM_$_mwh'][MeOH_scenario]
discount_rate = finance['MeOH']['discount_rate']
TASC_multiplier = finance['MeOH']['TASC_multiplier']
lcom_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,TASC_multiplier,discount_rate)
finance['MeOH']['lcom_$_kwh'] = {MeOH_scenario:lcom_kwh}
lcom_kg = list(np.multiply(lcom_kwh,MeOH_LHV_MJ_kg/3600*1000))
finance['MeOH']['lcom_$_kg'] = lcom_kg


# Print prices per unit for ALL scenarios
prices_to_check = ['OCC_$_kw','FOM_$_kwyr','VOM_$_mwh']
plants_to_check = plant_scenarios.keys()
for plant in plants_to_check:
    if plant == 'H2':
        scenario_list = H2A_scenarios
    elif plant == 'MeOH':
        scenario_list = MeOH_scenarios
    else:
        scenario_list = atb_scenarios
    for scenario in scenario_list:
        for price in prices_to_check:
            try:
                price_value = finance[plant][price]
                if type(price_value) is dict:
                    price_value = price_value[scenario]
                    if type(price_value) is list:
                        price_value = price_value[0]
                print('{} plant {} ({} scenario): ${}'.format(plant,
                                                                price,
                                                                scenario,
                                                                price_value))
            except:
                print('{} plant did not import {} for {} scenario'.format(plant,
                                                                        price,
                                                                        scenario))

# Print absolute prices and parameters for specific scenarios
prices_to_check = ['OCC_$','FOM_$_yr','VOM_$_yr']
params_to_check = ['output_kw'] 
for plant in plants_to_check:
    scenario = plant_scenarios[plant]
    for price in prices_to_check:
        try:
            price_value = finance[plant][price]
            if type(price_value) is dict:
                price_value = price_value[scenario]
                if type(price_value) is list:
                    price_value = price_value[0]
            print('{} plant {} ({} scenario): ${}'.format(plant,
                                                            price,
                                                            scenario,
                                                            price_value))
        except:
            print('{} plant did not import {} for {} scenario'.format(plant,
                                                                    price,
                                                                    scenario))
    for param in params_to_check:
        try:
            param_value = engin[plant][param]
            if type(param_value) is dict:
                param_value = param_value[scenario]
                if type(param_value) is list:
                    param_value = param_value[0]
            print('{} plant {} ({} scenario): {}'.format(plant,
                                                            param,
                                                            scenario,
                                                            param_value))
        except:
            print('{} plant did not import {} for {} scenario'.format(plant,
                                                                    param,
                                                                    scenario))

##