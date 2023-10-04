import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def calc_lcoe(OCC_kw, FOM_kwyr, VOM_mwh, Cf, TASC_multiplier, discount_rate):
    
    for arg in [OCC_kw, FOM_kwyr, VOM_mwh, Cf, TASC_multiplier, discount_rate]:
        arg = np.array(arg)
    
    TCC_kw = np.multiply(OCC_kw,TASC_multiplier)
    TCC_recovery_kwyr = np.multiply(TCC_kw,discount_rate)
    TFC_kwyr = np.add(TCC_recovery_kwyr,FOM_kwyr)
    TFC_kwh = np.divide(TFC_kwyr,8760)
    TFC_kwh = np.divide(TFC_kwh, Cf)
    VOM_kwh = np.divide(VOM_mwh,1000)
    lcoe_kwh = np.add(TFC_kwh,VOM_kwh)

    return lcoe_kwh


def try_H2_price(Forced_H2_Price, index, plotting=False, DAC_cost_mt=0, run_idx=0):

    Force_H2_1 = False
    Force_hyb_ems = False # Force hybrid emissions down to 0 gCO2e/MWh by 2050
    add_DAC = True

    ## Load dicts from json dumpfiles

    current_dir = Path(__file__).parent.absolute()
    ax_list = []
    line_styles = ['-','--',':']
    cambium_scenarios = ['MidCase',]#'LowNGPrice','HighNGPrice']#
    for l, cambium_scenario in enumerate(cambium_scenarios):
        style = line_styles[l]

        if l == 1:
            Force_H2_1 = True
        
        resource_dir = current_dir/'..'/'resource_files'/'methanol_RCC'
        results_dir = resource_dir/'HOPP_results'/cambium_scenario
        with open(Path(results_dir/'engin.json'),'r') as file:
            engin = json.load(file)
        with open(Path(results_dir/'finance.json'),'r') as file:
            finance = json.load(file)
        with open(Path(results_dir/'scenario.json'),'r') as file:
            scenario_info = json.load(file)
            plant_scenarios = scenario_info['plant_scenarios']
            # cambium_scenarios = scenario_info['cambium_scenarios']
            # cambium_scenario = scenario_info['cambium_scenario']
            atb_scenarios = scenario_info['atb_scenarios']
            H2A_scenarios = scenario_info['H2A_scenarios']
            MeOH_scenarios = scenario_info['MeOH_scenarios']
        with open(Path(results_dir/'locations.json'),'r') as file:
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

        site_name = 'TX09'#scenario_info['site_selection']['site_name']
        site_num = 1#scenario_info['site_selection']['site_num']
        location = {}
        for key, value in locations[site_name].items():
            if type(value) is not dict:
                location[key] = value[site_num-1]
            else:
                location[key] = {}
                for sub_key, sub_value in value.items():
                    location[key][sub_key] = sub_value[site_num-1]


        # Size renewable plants using location data

        h_plant = 'HCO2'
        pv_out_kw = location[h_plant]['pv_output_kw']
        pv_cap_kw = location[h_plant]['pv_capacity_kw']
        wind_out_kw = location[h_plant]['wind_output_kw']
        wind_cap_kw = location[h_plant]['wind_capacity_kw']
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
            Cf = engin[NG_plant]['capacity_factor']
            ng_lcoe_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,Cf,TASC_multiplier,discount_rate)
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
                VOM_CO2_mwh = list(np.divide(VOM_CO2_yr,np.multiply(output_kw,8.76)))
                if 'CO2' in plant:
                    finance[plant]['VOM_CO2_$_yr'] = {MeOH_scenario:VOM_CO2_yr}
                    finance[plant]['VOM_CO2_$_mwh'] = {MeOH_scenario:VOM_CO2_mwh}
                else:
                    finance[plant]['VOM_CO2_$_yr'] = {MeOH_scenario:0}
                    finance[plant]['VOM_CO2_$_mwh'] = {MeOH_scenario:list(np.zeros(finance[plant]['plant_lifespan']))}

        ## Get H2 production cost

        for plant in ['HCO2','HPSR']:
            OCC_kw = finance[plant]['OCC_$_kw'][h2_scenario]
            FOM_kwyr = finance[plant]['FOM_$_kwyr'][h2_scenario]
            VOM_mwh = finance[plant]['VOM_$_mwh'][h2_scenario]
            discount_rate = finance[plant]['discount_rate']
            TASC_multiplier = finance[plant]['TASC_multiplier']
            lcoh_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,1,TASC_multiplier,discount_rate)
            finance[plant]['lcoh_$_kwh'] = {h2_scenario:lcoh_kwh}
            H2_LHV_MJ_kg = engin[plant]['H2_LHV_MJ_kg']
            lcoh_kg = list(np.multiply(lcoh_kwh,H2_LHV_MJ_kg/3600*1000))
            finance[plant]['lcoh_$_kg'] = lcoh_kg
            finance['M'+plant[1:]]['VOM_H2_$_kgH2'] = {MeOH_scenario:lcoh_kg}
            finance['M'+plant[1:]]['VOM_NG_$_mwh'] = {}
            for meoh_scenario in MeOH_scenarios:
                finance['M'+plant[1:]]['VOM_NG_$_mwh'][meoh_scenario] = list(np.zeros(finance[plant]['plant_lifespan']))
        for plant in ['MSMR','MSMC']:
            finance[plant]['VOM_H2_$_mwh'] = {}
            finance[plant]['VOM_CO2_$_mwh'] = {}
            for meoh_scenario in MeOH_scenarios:
                finance[plant]['VOM_H2_$_mwh'][meoh_scenario] = list(np.zeros(finance[plant]['plant_lifespan']))
                finance[plant]['VOM_CO2_$_mwh'][meoh_scenario] = list(np.zeros(finance[plant]['plant_lifespan']))


        # Import TRACI impact analysis results for individual plants
        elec_rows = [0]
        elec_rows.extend(list(np.arange(28,38)))
        co2_rows = list(np.arange(0,28))
        co2_rows.extend(list(np.arange(31,38)))
        h2_rows = list(np.arange(0,31))
        h2_rows.extend(list(np.arange(34,38)))
        methanol_rows = list(np.arange(0,34))
        elec_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=elec_rows)
        elec_lca_df = elec_lca_df.iloc[:,:7]
        co2_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=co2_rows)
        co2_lca_df = co2_lca_df.iloc[:,:7]
        h2_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=h2_rows)
        h2_lca_df = h2_lca_df.iloc[:,:7]
        methanol_lca_df = pd.read_csv(resource_dir/'LCA_Inputs.csv', index_col=0, skiprows=methanol_rows)
        methanol_lca_df = methanol_lca_df.iloc[:,:7]

        # Get impact factor units
        factor_names = ['Eutrophication Potential',
                        'Acidification Potential',
                        'Particulate Matter Formation Potential',
                        'Photochemical Smog Formation Potential',
                        'Global Warming Potential',
                        'Ozone Depletion Potential',
                        'Water Consumption']
        traci_factors = {}
        traci_units = elec_lca_df.columns.values
        for i, name in enumerate(factor_names):
            traci_factors[name] = traci_units[i][:-4]

        # Interpolate grid LCA for site over 2020-2050 from 5 year intervals
        grid_lca = pd.DataFrame(np.zeros((31,7)),index=np.arange(2020,2051),columns=traci_units)
        five_yrs = np.arange(20,sim_years[-1]+5-2000,5)
        for five_yr in five_yrs:
            df_index = site_name[:2]+str(five_yr)
            grid_lca.loc[five_yr+2000] = elec_lca_df.loc[df_index]
            if five_yr > 20:
                for i, year in enumerate(np.arange(five_yr-4,five_yr)):
                    prev_lca = grid_lca.loc[five_yr-5+2000].values
                    next_lca = grid_lca.loc[five_yr+2000].values
                    year_lca = np.zeros(len(factor_names))
                    for j in range(len(factor_names)):
                        year_lca[j] = prev_lca[j] + (next_lca[j]-prev_lca[j])*(i+1)/5
                    grid_lca.loc[year+2000] = year_lca

        # Put lca prices in dict
        lca = {}
        lca['Grid'] = {}
        elec_techs = ['NGCC','CCS','PV','LBW','OSW']
        for plant in elec_techs:
            lca[plant] = {}
            for unit in elec_lca_df.columns.values:
                lca[plant][unit] = elec_lca_df.loc[plant,unit]
                grid_lca_list = []
                for year in sim_years:
                    grid_lca_list.append(grid_lca.loc[year,unit])
                lca['Grid'][unit] = grid_lca_list
        for plant in ['NGCC','CCS']:
            for unit in co2_lca_df.columns.values:
                lca[plant][unit] = co2_lca_df.loc[plant,unit]
        for plant in ['H2St','H2NG']:
            lca[plant] = {}
            for unit in h2_lca_df.columns.values:
                lca[plant][unit] = h2_lca_df.loc[plant,unit]
        for plant in ['MeRe','MeNG','MeHy']:
            lca[plant] = {}
            for unit in methanol_lca_df.columns.values:
                lca[plant][unit] = methanol_lca_df.loc[plant,unit]

        # Workaround for getting ASPEN sweep CO2/WC values in
        lca['MeRe']['kgCO2e_kgMeOH'] = engin['MPSR']['CO2_em_kg_MeOH']
        lca['MeRe']['kgH2O_kgMeOH'] = engin['MPSR']['H2O_kg_yr_in'][MeOH_scenario]/engin['MPSR']['MeOH_kg_yr'][MeOH_scenario]


        lca['CO2'] = {}
        lca['H2'] = {}
        lca['MeOH'] = {}
        lca['orig_elec'] = {}
        lca['elec'] = {}
        lca['stack'] = {}
        # Calculate impact factors per unit methanol
        plants = ['MSMR','MSMC','MCO2','MPSR']#
        for m, me_plant in enumerate(plants):
            lca[me_plant] = {}
            for unit in lca['MeNG'].keys():
            
                MeOH_out = engin[me_plant]['MeOH_kg_yr'][MeOH_scenario]
                
                # CO2 production - impact added by CCS plant over NGCC plant (CCS captured CO2)
                # OR zero impact (NGCC flue gas) - NGCC flue gas impact attributed to ELECTRICITY out to grid
                CO2_in = engin[me_plant]['CO2_kg_yr_in'][MeOH_scenario]
                if 'SMC' in me_plant:
                    no_cc_CO2 = engin[me_plant]['TS_CO2_kg_yr'][MeOH_scenario]
                    if 'CO2e' in unit:
                        lca['CO2'][unit] = -no_cc_CO2/MeOH_out
                elif 'CO2' in me_plant:
                    CCS_CO2_kgMeOH = lca['CCS'][unit[:-4]+'CO2']*CO2_in/MeOH_out
                    NGCC_CO2_kgMeOH = lca['NGCC'][unit[:-4]+'CO2']*CO2_in/MeOH_out
                    lca['CCS'][unit] = CCS_CO2_kgMeOH-NGCC_CO2_kgMeOH
                    lca['CO2'][unit] = CCS_CO2_kgMeOH-NGCC_CO2_kgMeOH
                else:
                    lca['NGCC'][unit] = 0
                    lca['CO2'][unit] = 0
                
                if 'SM' in me_plant:
                    lca['stack'][unit] = 0
                elif 'CO2' in unit:
                    stack_CO2 = engin[me_plant]['Stack_CO2_kg_yr'][MeOH_scenario]
                    stack_CO2_kgMeOH = stack_CO2/MeOH_out
                    lca['stack'][unit] = stack_CO2_kgMeOH
                else:
                    lca['stack'][unit] = 0
                
                # H2 production - hybrid plant + grid electricity
                H2_kg_in = engin[me_plant]['H2_kg_yr_in'][MeOH_scenario]
                H2st_kgMeOH = lca['H2St'][unit[:-4]+'H2']*H2_kg_in/MeOH_out
                lca['H2St'][unit] = H2st_kgMeOH
                if ('CO2' in unit) and ('PSR' in me_plant):
                    print(H2_kg_in)
                # pv_kw_out = locations[site_name]['pv_output_kw'][site_num-1]
                # wind_kw_out = locations[site_name]['wind_output_kw'][site_num-1]
                wind_plant = 'LBW' if location['on_land'] else 'OSW'
                for plant in ['PV',wind_plant,'Grid','elec','orig_elec','H2']:
                    lca[plant][unit[:-4]+'H2'] = []
                    lca[plant][unit] = []
                    if 'elec' in plant:
                        lca[plant][unit[:-6]+'MWh'] = []
                for i, year in enumerate(sim_years):
                    # pct_pv = 100*pv_kw_out[i]/(pv_kw_out[i]+wind_kw_out[i])
                    # pct_wind = 100-pct_pv
                    if 'SM' in me_plant:
                        H2_MWh_yr = 0
                    else:
                        H2_MWh_yr = H2_kg_in*engin['H'+me_plant[1:]]['elec_use_kwh_kgH2']['Future'][i]/1000   #location['H'+me_plant[1:]]['electrolyzer_input_kw'][i]*8.76
                    # pv_em_MWh = lca['PV'][unit[:-6]+'MWh']*pct_pv/100
                    # wind_em_MWh = lca[wind_plant][unit[:-6]+'MWh']*pct_wind/100
                    # hyb_em_MWh = pv_em_MWh+wind_em_MWh
                    # grid_em_MWh = grid_lca.loc[year,unit[:-6]+'MWh']
                    # buy_em_MWh = grid_em_MWh-hyb_em_MWh
                    # sell_em_MWh = hyb_em_MWh-grid_em_MWh
                    # grid_bought_MWh_yr = location[plant]['grid_bought_kw'][i]*8.76
                    # grid_sold_MWh_yr = location[plant]['grid_sold_kw'][i]*8.76
                    # buy_em_kg_yr = buy_em_MWh*grid_bought_MWh_yr
                    # sell_em_kg_yr = sell_em_MWh*grid_sold_MWh_yr
                    # elyzer_hyb_MWh_yr = H2_MWh_yr-grid_bought_MWh_yr
                    # lca['PV'][unit[:-4]+'H2'].append(elyzer_hyb_MWh_yr*pct_pv/100*hyb_em_MWh/H2_kg_in)
                    # lca[wind_plant][unit[:-4]+'H2'].append(elyzer_hyb_MWh_yr*pct_wind/100*hyb_em_MWh/H2_kg_in)
                    # lca['Grid'][unit[:-4]+'H2'].append((buy_em_kg_yr+sell_em_kg_yr)/H2_kg_in)
                    if ('CO2' in unit) or ('H2O' in unit):
                        if 'CO2' in unit:
                            em_power_unit = 'CI_g_kwh'
                            orig_em_power_unit = 'orig_CI_g_kwh'
                            # if 'CO2' in me_plant:
                            #     if i == 0:
                            #         lca['MeHy'][unit] = []
                            #     grid_ghgs_kg_mwh = [118.3726,69.42419,50.95484,46.73548,45.39355,44.86129,44.6]
                            #     lca['MeHy'][unit].append(engin[me_plant]['elec_in_kw'][MeOH_scenario]*8.76*grid_ghgs_kg_mwh[i]/MeOH_out)
                            # elif 'PSR' in me_plant:
                            #     if i == 0:
                            #         lca['MeRe'][unit] = []
                            #     grid_ghgs_kg_mwh = [118.3726,69.42419,50.95484,46.73548,45.39355,44.86129,44.6]
                            #     lca['MeRe'][unit].append(engin[me_plant]['elec_in_kw'][MeOH_scenario]*8.76*grid_ghgs_kg_mwh[i]/MeOH_out)
                        else:
                            em_power_unit = 'WC_g_kwh'
                            orig_em_power_unit = 'orig_WC_g_kwh'
                        if 'SM' not in me_plant:
                            elec_em_kg_MWh = location['H'+me_plant[1:]][em_power_unit][i]
                            orig_elec_em_kg_MWh = location['H'+me_plant[1:]][orig_em_power_unit][i]
                            if (l == 0) and Force_hyb_ems:
                                max_elec_em = location['H'+me_plant[1:]][em_power_unit][-1]
                                elec_em_kg_MWh = (0/max_elec_em*elec_em_kg_MWh*i+elec_em_kg_MWh*(len(sim_years)-i-1))/(len(sim_years)-1)
                                orig_elec_em_kg_MWh = (0/max_elec_em*orig_elec_em_kg_MWh*i+orig_elec_em_kg_MWh*(len(sim_years)-i-1))/(len(sim_years)-1)
                        elif 'SMC' in me_plant:
                            # Adjust by carbon intensity of adding carbon capture to NGCC plant (adjusted for power)
                            CCS_CO2_kgCO2 = lca['CCS'][unit[:-4]+'CO2']
                            CCS_CO2_MWh = CCS_CO2_kgCO2*engin['CCS']['CO2_out_kg_mwh']
                            NGCC_CO2_kgCO2 = lca['NGCC'][unit[:-4]+'CO2']
                            NGCC_CO2_MWh = NGCC_CO2_kgCO2*engin['NGCC']['CO2_out_kg_mwh']
                            elec_out_mwh_yr = engin[me_plant]['elec_out_mwh_yr'][MeOH_scenario]
                            cc_increase_co2e_kg_yr = elec_out_mwh_yr*(CCS_CO2_MWh-NGCC_CO2_MWh)
                            meoh_kg_yr = engin[me_plant]['MeOH_kg_yr'][MeOH_scenario]
                            kg_co2_kg_meoh = cc_increase_co2e_kg_yr/meoh_kg_yr
                            lca['CO2'][unit] += kg_co2_kg_meoh
                        else:
                            elec_em_kg_MWh = 0
                            orig_elec_em_kg_MWh = 0
                        elec_em_kg_yr = elec_em_kg_MWh*H2_MWh_yr
                        if H2_kg_in > 0:
                            elec_em_kg_H2 = elec_em_kg_yr/H2_kg_in
                        else:
                            elec_em_kg_H2 = 0
                        lca['elec'][unit[:-6]+'MWh'].append(elec_em_kg_MWh)
                        lca['elec'][unit[:-4]+'H2'].append(elec_em_kg_H2)
                        lca['orig_elec'][unit[:-6]+'MWh'].append(orig_elec_em_kg_MWh)
                    else:
                        lca['elec'][unit[:-6]+'MWh'].append(0)
                        lca['elec'][unit[:-4]+'H2'].append(0)
                        lca['orig_elec'][unit[:-6]+'MWh'].append(0)
                    lca['H2'][unit[:-4]+'H2'].append(lca['elec'][unit[:-4]+'H2'][i] +
                                                    lca['H2St'][unit[:-4]+'H2'])
                    for plant in ['elec','H2']:
                        lca[plant][unit].append(lca[plant][unit[:-4]+'H2'][i]*H2_kg_in/MeOH_out)

                # Add together components of methanol lca: H2 in, CO2 in, and reactor
                lca[me_plant][unit] = []
                for i, year in enumerate(sim_years):
                    if 'SM' in me_plant:
                        lca[me_plant][unit].append(lca['CO2'][unit] + lca['MeNG'][unit])
                    elif 'CO2' in me_plant:
                        if 'CO2' in unit:
                            lca[me_plant][unit].append(lca['CO2'][unit] + lca['H2'][unit][i] + lca['MeHy'][unit])#[i]) # + lca['stack'][unit]
                        else:
                            lca[me_plant][unit].append(lca['CO2'][unit] + lca['H2'][unit][i] + lca['MeHy'][unit])
                    else:
                        if 'CO2' in unit:
                            lca[me_plant][unit].append(lca['CO2'][unit] + lca['H2'][unit][i] + lca['MeRe'][unit])#[i]) # + lca['stack'][unit]
                        else:
                            lca[me_plant][unit].append(lca['CO2'][unit] + lca['H2'][unit][i] + lca['MeRe'][unit])
                # Break down lca into components:
                if ('CO2' in unit) or ('H2O' in unit):
                    lca[me_plant][unit+'_hyb_elec'] = []
                    lca[me_plant][unit+'_grid_elec_disp'] = []
                    lca[me_plant][unit+'_elyzer'] = []
                    lca[me_plant][unit+'_reactor'] = []
                    lca[me_plant][unit+'_ccs'] = []
                    for i, year in enumerate(sim_years):
                        if 'SM' not in me_plant:
                            lca[me_plant][unit+'_hyb_elec'].append(lca['orig_elec'][unit[:-6]+'MWh'][i]/MeOH_out*H2_MWh_yr)
                            lca[me_plant][unit+'_grid_elec_disp'].append((lca['orig_elec'][unit[:-6]+'MWh'][i]-lca['elec'][unit[:-6]+'MWh'][i])/MeOH_out*H2_MWh_yr)
                            lca[me_plant][unit+'_elyzer'].append(lca['H2St'][unit[:-4]+'H2']/MeOH_out*H2_kg_in)
                            if 'CO2' in me_plant:
                                if 'CO2' in unit:
                                    lca[me_plant][unit+'_reactor'].append(lca['MeHy'][unit])#[i])
                                else:
                                    lca[me_plant][unit+'_reactor'].append(lca['MeHy'][unit])
                                lca[me_plant][unit+'_ccs'].append(lca['CCS'][unit])
                            else:
                                if 'CO2' in unit:
                                    lca[me_plant][unit+'_reactor'].append(lca['MeRe'][unit])#[i])
                                else:
                                    lca[me_plant][unit+'_reactor'].append(lca['MeRe'][unit])
                                lca[me_plant][unit+'_ccs'].append(0)
        
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
                if plant == 'MPSR':
                    Plant_Forced_H2_Price = Forced_H2_Price
                else:
                    Plant_Forced_H2_Price = Forced_H2_Price-finance['MPSR']['VOM_H2_$_kgH2'][MeOH_scenario][-1]+\
                                                            finance[plant]['VOM_H2_$_kgH2'][MeOH_scenario][-1]
                if Force_H2_1:
                    for i in range(len(H2_price_kg)):
                        H2_price_kg[i] = (Plant_Forced_H2_Price/H2_price_kg[-1]*H2_price_kg[i]*i+H2_price_kg[i]*(len(H2_price_kg)-i-1))/(len(H2_price_kg)-1)
                    finance[plant]['VOM_H2_$_kgH2'][MeOH_scenario] = H2_price_kg
                VOM_H2_yr = list(np.multiply(H2_price_kg,H2_kg_yr))
                finance[plant]['VOM_H2_$_yr'] = {MeOH_scenario:VOM_H2_yr}
                VOM_H2_mwh = [i/mwh_yr for i in VOM_H2_yr]
                finance[plant]['VOM_H2_$_mwh'] = {MeOH_scenario:VOM_H2_mwh}
                VOM_comps = ['VOM_H2_$_mwh',
                            'VOM_CO2_$_mwh',
                            'VOM_H2O_$_mwh',
                            # 'VOM_elec_$_mwh',
                            'VOM_cat_$_mwh',
                            'VOM_other_$_mwh']
            finance[plant]['VOM_$_mwh'] = {MeOH_scenario:[]}
            finance[plant]['VOM_$_yr'] = {MeOH_scenario:[]}
            finance[plant]['OCC_kg'] = {}
            finance[plant]['FOM_kg'] = {}
            finance[plant]['VOM_H2_kg'] = {}
            finance[plant]['VOM_CO2_kg'] = {}
            finance[plant]['VOM_NG_kg'] = {}
            finance[plant]['VOM_cat_kg'] = {}
            finance[plant]['VOM_kg'] = {}
            finance[plant]['VOM_other_kg'] = {}
            finance[plant]['VOM_DAC_kg'] = {}
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
            VOM_NG_mwh = finance[plant]['VOM_NG_$_mwh'][MeOH_scenario]
            VOM_H2_mwh = finance[plant]['VOM_H2_$_mwh'][MeOH_scenario]
            VOM_CO2_mwh = finance[plant]['VOM_CO2_$_mwh'][MeOH_scenario]
            VOM_cat_mwh = finance[plant]['VOM_cat_$_mwh'][MeOH_scenario]
            discount_rate = finance[plant]['discount_rate']
            TASC_multiplier = finance[plant]['TASC_multiplier']
            lcom_kwh = calc_lcoe(OCC_kw,FOM_kwyr,VOM_mwh,1,TASC_multiplier,discount_rate)
            finance[plant]['lcom_$_kwh'] = {MeOH_scenario:lcom_kwh}
            lcom_kg = list(np.multiply(lcom_kwh,MeOH_LHV_MJ_kg/3600*1000))
            VOM_DAC_kg = [i*DAC_cost_mt/1000 for i in lca[plant]['kgCO2e_kgMeOH']]
            finance[plant]['VOM_DAC_kg'][MeOH_scenario] = VOM_DAC_kg
            if add_DAC:
                lcom_kg = list(np.add(lcom_kg,VOM_DAC_kg))
            finance[plant]['lcom_$_kg'] = lcom_kg
            meoh_kg_yr = engin[plant]['MeOH_kg_yr'][MeOH_scenario]
            OCC_kg = list(np.multiply(calc_lcoe(OCC_kw,0,[0]*len(VOM_mwh),1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            FOM_kg = list(np.multiply(calc_lcoe(0,FOM_kwyr,[0]*len(VOM_mwh),1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            VOM_H2_kg = list(np.multiply(calc_lcoe(0,0,VOM_H2_mwh,1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            VOM_CO2_kg = list(np.multiply(calc_lcoe(0,0,VOM_CO2_mwh,1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            VOM_NG_kg = list(np.multiply(calc_lcoe(0,0,VOM_NG_mwh,1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            VOM_cat_kg = list(np.multiply(calc_lcoe(0,0,[VOM_cat_mwh]*len(VOM_mwh),1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            VOM_kg = list(np.multiply(calc_lcoe(0,0,VOM_mwh,1,TASC_multiplier,discount_rate),MeOH_LHV_MJ_kg/3600*1000))
            if add_DAC:
                VOM_kg = list(np.add(VOM_kg,VOM_DAC_kg))
            VOM_other_kg = [VOM_kg[i]-VOM_H2_kg[i]-VOM_CO2_kg[i]-VOM_NG_kg[i]-VOM_cat_kg[i] for i in range(len(VOM_kg))]
            if add_DAC:
                VOM_other_kg = [VOM_other_kg[i]-VOM_DAC_kg[i] for i in range(len(VOM_kg))]
            finance[plant]['OCC_kg'][MeOH_scenario] = OCC_kg
            finance[plant]['FOM_kg'][MeOH_scenario] = FOM_kg
            finance[plant]['VOM_H2_kg'][MeOH_scenario] = VOM_H2_kg
            finance[plant]['VOM_CO2_kg'][MeOH_scenario] = VOM_CO2_kg
            finance[plant]['VOM_NG_kg'][MeOH_scenario] = VOM_NG_kg
            finance[plant]['VOM_cat_kg'][MeOH_scenario] = VOM_cat_kg
            finance[plant]['VOM_kg'][MeOH_scenario] = VOM_kg
            finance[plant]['VOM_other_kg'][MeOH_scenario] = VOM_other_kg


        for m, me_plant in enumerate(plants):
            
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


            # # Print absolute prices and parameters for specific scenarios
            # prices_to_check = ['OCC_$','FOM_$_yr','VOM_$_mwh','VOM_other_$_mwh','VOM_elec_$_mwh','VOM_H2_$_mwh','VOM_CO2_$_mwh','VOM_NG_$_mwh','VOM_TS_$_mwh','VOM_H2O_$_mwh','VOM_cat_$_mwh']
            # params_to_check = ['output_kw','elec_out_mwh_yr','H2_kg_yr','CO2_kg_yr','MeOH_kg_yr','NG_kg_yr'] 
            # for plant in plants_to_check:
            #     scenario = plant_scenarios[plant]
            #     for price in prices_to_check:
            #         try:
            #             price_value = finance[plant][price]
            #             if type(price_value) is dict:
            #                 price_value = price_value[scenario]
            #                 if type(price_value) is list:
            #                     price_value = price_value[0]
            #             print('{} plant {} ({} scenario): ${:,.2f}'.format(plant,
            #                                                             price,
            #                                                             scenario,
            #                                                             price_value))
            #         except:
            #             print('{} plant did not import {} for {} scenario'.format(plant,
            #                                                                     price,
            #                                                                     scenario))
            #     for param in params_to_check:
            #         # if 'prod' in param:
            #         #     if 'CC' in plant:
            #         #         product = 'CO2'
            #         #     elif plant[0] == 'H':
            #         #         product = 'H2'
            #         #     elif plant[0] == 'M':
            #         #         product = 'MeOH'
            #         #     else:
            #         #         product = 'elec'
            #         #     if product == 'elec':
            #         #         param = 'elec_out_mwh_yr'
            #         #     else:
            #         #         param = param = product+'_kg_yr'
            #         try:
            #             possible_params = list(engin[plant].keys())
            #             for possible_param in possible_params:
            #                 if possible_param.find(param)>=0:
            #                     param = possible_param
            #             param_value = engin[plant][param]
            #             if type(param_value) is dict:
            #                 param_value = param_value[scenario]
            #                 if type(param_value) is list:
            #                     param_value = param_value[0]
            #             print('{} plant {} ({} scenario): {:,.2f}'.format(plant,
            #                                                             param,
            #                                                             scenario,
            #                                                             param_value))
            #         except:
            #             print('{} plant did not import {} for {} scenario'.format(plant,
            #                                                                     param,
            #                                                                     scenario))

            if plotting:

                labels = ['NGCC w/o carbon capture',
                        'NGCC with carbon capture',
                        'Wind/solar w/o grid exchange',
                        'Wind/solar with grid exchange']

                plots = [lcoe_ngcc,lcoe_ccs,orig_lcoe_kwh,lcoe_kwh]

                colors = [[0,0,0],
                        [0,0,1],
                        [0,0.5,0],
                        [1,0,0],
                        ]

                if l == 0:
                    ax = plt.subplot(2,3,1)
                    ax_list.append(ax)
                else:
                    plt.sca(ax_list[0])
                for i, label in enumerate(labels):
                    color = colors[i]
                    if l>0:
                        label = None
                    # if (m == 0) and (i > 1):
                    #     plt.plot(sim_years,plots[i],style,label=label,color=color)
                # plt.legend()
                plt.xlabel('Plant Startup Year')
                plt.ylabel('Levelized Cost [2020 $/kWh]')
                plt.title('Electricity Cost')
                plt.ylim([0,0.06])

                labels = ['CO2 captured from NGCC plant',
                        'H2 produced from electrolyzer',
                        'Methanol produced by Nyari et al. plant']
                
                labels = ['Mid-case','High NG Price','High NG Price + $1 Green H2 by 2050']

                label = labels[l]
                if m>0:
                    label = None
                plt.plot([0,1],[0,1],style,color=[0,0,0],label=label)
                plt.legend()

                labels = ['Baseline #0: SMR without Carbon Capture',
                        'Baseline #1: SMR with Carbon Capture',
                        'Baseline #2: CO2 Hydrogenation State of Art',
                        'Novel Technology: NREL PSRs, H2:MeOH Ratio = {:.2f}']


                mass_ratio_CO2_MeOH = 1.397 # kg CO2 in needed per kg of MeOH produced
                mass_ratio_H2_MeOH = 0.199 # kg H2 in needed per kg of MeOH produced
                lcoh_kg = [i*mass_ratio_H2_MeOH for i in lcoh_kg]
                CO2_price_kg = [i*mass_ratio_CO2_MeOH for i in CO2_price_kg]

                plots = [CO2_price_kg,lcoh_kg,lcom_kg]
                # plants = ['MSMR','MSMC','MCO2','MPSR']

                label = labels[m]
                if m==3:
                    label = label.format(0.29-0.04*index)

                if l == 0:
                    ax = plt.subplot(2,3,2)
                    ax_list.append(ax)
                else:
                    plt.sca(ax_list[1])
                color = colors[m]
                if l>0:
                    label = None
                lcom = finance[me_plant]['lcom_$_kg']
                if m > 0:
                    # if (index == 0) or (m == 3):
                    plt.plot(sim_years,lcom,style,label=label,color=color)
                # plt.legend()
                plt.xlabel('Plant Startup Year')
                plt.ylabel('Levelized Cost [2020 $/kg of Methanol]')
                plt.title('Methanol Cost')
                plt.ylim([0,2])
                plt.grid('on',axis='both')
                # if index == 2:
                #     plt.legend()
                
                if l == 0:
                    ax = plt.subplot(2,3,3)
                    ax_list.append(ax)
                else:
                    plt.sca(ax_list[2])
                color = colors[m]
                if l>0:
                    label = None
                if m > 2:
                    lcoh = finance[me_plant]['VOM_H2_$_kgH2'][MeOH_scenario]
                    plt.plot(sim_years,lcoh,style,label=labels[m],color=[0,0,0])#colors[m])
                    # plt.legend()
                    plt.xlabel('Plant Startup Year')
                    plt.ylabel('Levelized Cost [2020 $/kg of Hydrogen]')
                    plt.title('Green Hydrogen Cost')
                plt.ylim([0,4])
                plt.grid('on',axis='both')

                labels = ['NGCC w/o carbon capture',
                        'NGCC with carbon capture',
                        'Wind/solar w/o grid exchange',
                        'Wind/solar with grid exchange']

                label = labels[m]
                
                if l == 0:
                    ax = plt.subplot(2,3,4)
                    ax_list.append(ax)
                else:
                    plt.sca(ax_list[3])
                for i, label in enumerate(labels):
                    color = colors[i]
                    if l>0:
                        label = None
                elec_em = copy.deepcopy(lca['elec']['kgCO2e_MWh'])
                orig_elec_em = copy.deepcopy(lca['orig_elec']['kgCO2e_MWh'])
                # if 'PSR' in me_plant:
                #     plt.plot(sim_years,orig_elec_em,style,label=labels[m-1],color=colors[m-1])
                #     plt.plot(sim_years,elec_em,style,label=labels[m],color=colors[m])
                # plt.plot(sim_years,lca['MeOH']['kgCO2e_kgMeOH'],label='Methanol from NGCC-captured CO2 + Green Hydrogen')
                plt.xlabel('Plant Startup Year')
                plt.ylabel('kg CO2-equivalent / MWh_e')
                plt.title('Electricity LCA')
                # plt.legend()
                plt.ylim([0,65])
                # plt.grid(axis='both')
                
                labels = ['Baseline #0: SMR without Carbon Capture',
                        'Baseline #1: SMR with Carbon Capture',
                        'Baseline #2: CO2 Hydrogenation State of Art',
                        'Novel Technology: NREL PSRs']
                
                labels = ['Baseline #0: SMR without Carbon Capture',
                        'Baseline #1: SMR with Carbon Capture',
                        'Baseline #2: CO2 Hydrogenation w/Green H2',
                        'Novel Technology: NREL PSRs']
                
                label = labels[m]
                if l>0:
                    label = None
                if m>0:
                    plt.plot([0,1],[0,1],style,label=label,color=colors[m])
                plt.legend()

                if l == 0:
                    ax = plt.subplot(2,3,5)
                    ax_list.append(ax)
                else:
                    plt.sca(ax_list[4])
                color = colors[m]
                if l>0:
                    label = None
                if m > 0:
                    plt.plot(sim_years,lca[me_plant]['kgCO2e_kgMeOH'],style,label=label,color=color)
                # plt.plot(sim_years,lca['MeOH']['kgCO2e_kgMeOH'],label='Methanol from NGCC-captured CO2 + Green Hydrogen')
                plt.xlabel('Plant Startup Year')
                plt.ylabel('kg CO2-equivalent / kg methanol')
                plt.title('Methanol LCA')
                # plt.legend()
                plt.ylim([0,1.2])
                plt.grid('on',axis='both')

                if l == 0:
                    ax = plt.subplot(2,3,6)
                    ax_list.append(ax)
                else:
                    plt.sca(ax_list[5])
                color = colors[m]
                if l>0:
                    label = None
                if m > 1:
                    lcom = finance[me_plant]['lcom_$_kg']
                    lcom_baseline = finance['MSMC']['lcom_$_kg']
                    ci = lca[me_plant]['kgCO2e_kgMeOH']
                    ci_baseline = lca['MSMC']['kgCO2e_kgMeOH']
                    breakeven_co2_price = np.divide((np.subtract(lcom,lcom_baseline)),(np.subtract(ci_baseline,ci)))
                    # if m == 2:
                    #     breakeven_co2_price[0] = np.nan
                    plt.plot(sim_years,breakeven_co2_price,style,label=labels[m],color=colors[m])
                    # plt.legend()
                    plt.xlabel('Plant Startup Year')
                    plt.ylabel('CO2 price [2020 $/kg of CO2e emitted]')
                    plt.title('Breakeven CO2 Emissions Price')
                plt.ylim([0,1])
                plt.grid('on',axis='both')

                
                
                plt.gcf().set_tight_layout(True)

    if plotting:
        plt.show()

    print(finance['MCO2']['lcom_$_kg'])
    print(finance['MPSR']['lcom_$_kg'])
    print(finance['MPSR']['VOM_H2_$_kgH2'][MeOH_scenario])

    out_name = 'output_{}_{}_'.format(DAC_cost_mt,run_idx)#input("Name of output file:")

    out_plants = ['MCO2','MPSR']
    out_rows = ['lcom_$_kg','OCC_kg','FOM_kg','VOM_kg','VOM_H2_kg','VOM_CO2_kg','VOM_NG_kg','VOM_cat_kg','VOM_DAC_kg','VOM_other_kg',
                'kgCO2e_kgMeOH','kgCO2e_kgMeOH_hyb_elec','kgCO2e_kgMeOH_grid_elec_disp','kgCO2e_kgMeOH_elyzer','kgCO2e_kgMeOH_reactor','kgCO2e_kgMeOH_ccs',
                'kgH2O_kgMeOH','kgH2O_kgMeOH_hyb_elec','kgH2O_kgMeOH_grid_elec_disp','kgH2O_kgMeOH_elyzer','kgH2O_kgMeOH_reactor','kgH2O_kgMeOH_ccs']
    for i, year in enumerate(sim_years):
        out_frame = pd.DataFrame(np.zeros((len(out_rows),len(out_plants))),index=out_rows,columns=out_plants)
        for plant in out_plants:
            for row in out_rows:
                if ('CO2e' in row) or ('H2O' in row):
                    if '_kgH2' in row:
                        if plant is 'MSMC':
                            data = [0.0*i for i in lca[plant]['kgH2O_kgMeOH']]
                        else:
                            em_meoh = lca[plant]['kgH2O_kgMeOH']
                            data = [i*engin[plant]['MeOH_kg_yr'][MeOH_scenario]/engin[plant]['H2_kg_yr_in'][MeOH_scenario] for i in em_meoh]
                    else:
                        data = lca[plant][row]
                        if '_disp' in row:
                            data = [-i for i in data]
                else:
                    data = finance[plant][row]
                if type(data) is dict:
                    data = data[MeOH_scenario]
                if type(data) is list:
                    data = data[i]
                out_frame.loc[row,plant] = data
        out_frame.to_csv(Path('C:/Users/jmartin4/Documents/Projects/22 CO2 to Methanol')/(out_name+str(year)+'.csv'))

    return (finance['MCO2']['lcom_$_kg'][-1]-finance['MPSR']['lcom_$_kg'][-1])