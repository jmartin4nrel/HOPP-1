import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from hopp.simulation.technologies.financial.simple_financial_model import inflate

wind_ppa_lcoe_ratio = 0.7742
solar_ppa_lcoe_ratio = 0.6959

all_grid_wcs = [1718.605915	,1576.131286	,1433.656658	,1291.182029	,1148.707401	,
                1006.232772	,877.3949737	,748.5571751	,619.7193766	,490.881578	,
                362.0437794	,331.9507828	,301.8577862	,271.7647896	,241.671793	,
                211.5787964	,204.5211937	,197.463591     ,190.4059883	,183.3483856	,
                176.2907829	,175.0760655	,173.861348	    ,172.6466306	,171.4319132	,
                170.2171957	,167.6368848	,165.0565739	,162.476263	    ,159.8959522	,157.3156413]

def extract_cambium_data(cambium_dir, dollar_year):

    resource_dir = Path(__file__).parent.absolute()/'inputs'

    scenario_names = {'Mid-case':'MidCase',
                'High natural gas prices':'HighNGPrice',
                'Low natural gas prices':'LowNGPrice'}

    cont_48 = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN", 
            "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", 
            "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", 
            "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    grid_mix_key = {'Biomass':'biomass',
                    'Coal':'coal',
                    'Natural gas':'gas-ct',
                    'Geothermal':'geothermal',
                    'Hydro':'hydro',
                    'Nuclear':'nuclear',
                    'Petroleum':'o-g-s',
                    'Solar':['distpv','upv'],
                    'Wind':['wind-ons','wind-ofs'],
                    'NGCC':'gas-cc',
                    'NGCC w/ CCS':'gas-cc-ccs'}

    prefix = {2020:'StdScen20',
            2022:'StdScen21',
            2024:'Cambium22',
            2026:'Cambium22',
            2028:'Cambium22',
            2030:'Cambium22',
            2035:'Cambium22',
            2040:'Cambium22',
            2045:'Cambium22',
            2050:'Cambium22'}

    cambium_items = {'Natural Gas Price [2020$/MMBtu]':'repgasprice_nat.csv',
                'End Use Cost [2020$/MWh]':'total_cost_enduse',
                'Average Regional Generation Emissions [kgCO2e/MWh]':'aer_gen_co2e'}

    aeo_price_fn = 'Average_retail_price_of_electricity.csv'

    # Import AEO prices
    aeo_prices = pd.read_csv(cambium_dir/aeo_price_fn, skiprows=[0,1,2,3,5])
    aeo_keys = aeo_prices['source key'].values
    aeo_keys = [i[11:-6] for i in aeo_keys]
    index_check = []
    new_index = []
    for key in aeo_keys:
        is_state = key in cont_48
        index_check.append(is_state)
        if is_state:
            new_index.append(key)
    aeo_prices = aeo_prices.iloc[np.where(index_check)[0]]
    aeo_prices = aeo_prices.loc[:,['2020','2021','2022']]
    aeo_prices.index = new_index
    aeo_prices.columns = [int(i) for i in aeo_prices.columns]
    aeo_prices = aeo_prices.sort_index()
    for column in aeo_prices.columns.values:
        year_prices = aeo_prices.loc[:,column]
        year = int(column)
        aeo_prices.loc[:,column] = inflate(year_prices, year, dollar_year)

    # Import annual cambium data
    annual_cambium_prices = pd.DataFrame(None,index=new_index)
    annual_cambium_ghgs = pd.DataFrame(None,index=new_index)
    for year in [2020,2022]:
        fn = prefix[year]+'_'+scenario_names['Mid-case']+'_annual_state.csv'
        temp_df = pd.read_csv(cambium_dir/'Mid-case'/fn,skiprows=year-2018,index_col=0)
        temp_df = temp_df.loc[new_index]
        temp_df.reset_index(inplace=True)
        temp_df.set_index('t',inplace=True)
        temp_df = temp_df.loc[year]
        temp_price = temp_df.loc[:,'total_cost_enduse']
        if year == 2020:
            temp_ghg = temp_df.loc[:,'co2_rate_avg_gen']
        else:
            temp_ghg = temp_df.loc[:,'aer_gen_co2e']
        prices = inflate(temp_price.values, (year-2020)/2+2019, dollar_year)
        new_series = pd.Series(prices,new_index,name=year)
        annual_cambium_prices = annual_cambium_prices.join(new_series)
        new_series = pd.Series(temp_ghg.values,new_index,name=year)
        annual_cambium_ghgs = annual_cambium_ghgs.join(new_series)
    mid_prices = []
    mid_ghgs = []
    for state in new_index:
        values = annual_cambium_prices.loc[state].values
        mid_prices.append(np.mean(values))
        values = annual_cambium_ghgs.loc[state].values
        mid_ghgs.append(np.mean(values))
    new_series = pd.Series(mid_prices,new_index,name=2021)
    annual_cambium_prices = annual_cambium_prices.join(new_series)
    annual_cambium_prices = annual_cambium_prices.sort_index(axis=1)
    annual_cambium_prices = annual_cambium_prices.sort_index()
    new_series = pd.Series(mid_ghgs,new_index,name=2021)
    annual_cambium_ghgs = annual_cambium_ghgs.join(new_series)
    annual_cambium_ghgs = annual_cambium_ghgs.sort_index(axis=1)
    annual_cambium_ghgs = annual_cambium_ghgs.sort_index()

    # Correlate cambium data with AEO retail prices
    cambium = np.mean(annual_cambium_prices.values,axis=1)
    aeo = np.mean(aeo_prices.values,axis=1)*10 # cents/kWh to $/MWh
    state_multipliers = {}
    for i, state in enumerate(aeo_prices.index.values):
        state_multipliers[state] = aeo[i]/cambium[i]
    with open(Path(resource_dir/'cambium_aeo_multipliers.json'),'w') as file:
        json.dump(state_multipliers, file)

    # Add cambium prices for different scenarios
    key_years = [2022,2024,2026,2028,2030,2035,2040,2045,2050]
    years = np.arange(2023,2051)
    for scenario, name in scenario_names.items():
        cambium_prices = copy.deepcopy(annual_cambium_prices)
        cambium_ghgs = copy.deepcopy(annual_cambium_ghgs)
        cambium_ng = pd.read_csv(cambium_dir/scenario/'repgasprice_nat.csv',index_col=0)
        cambium_ng = cambium_ng.loc[2020:2050]
        ng2021 = pd.DataFrame(np.mean(cambium_ng.loc[2020:2022].values),index=[2021],columns=['Val'])
        cambium_ng = pd.concat([cambium_ng,ng2021])
        fn = 'Cambium22_'+name+'_annual_state.csv'
        temp_df = pd.read_csv(cambium_dir/scenario/fn,skiprows=5,index_col=0)
        temp_df = temp_df.loc[new_index]
        temp_df.reset_index(inplace=True)
        temp_df.set_index('t',inplace=True)
        for year in key_years[1:]:
            year_df = temp_df.loc[year]
            temp_price = year_df.loc[:,'total_cost_enduse']
            temp_ghg = year_df.loc[:,'aer_gen_co2e']
            prices = temp_price.values
            prices = inflate(prices, 2021, dollar_year)
            new_series = pd.Series(prices,new_index,name=year)
            cambium_prices = cambium_prices.join(new_series)
            new_series = pd.Series(temp_ghg.values,new_index,name=year)
            cambium_ghgs = cambium_ghgs.join(new_series)
        for year in years:
            if year not in key_years:
                key_idx = np.argmax([year<i for i in key_years])
                prev_key_year = key_years[key_idx-1]
                next_key_year = key_years[key_idx]
                interp_frac = (year-prev_key_year)/(next_key_year-prev_key_year)
                if prev_key_year == 2022:
                    fn = 'StdScen21_MidCase_annual_state.csv'
                    temp_prev_df = pd.read_csv(cambium_dir/'Mid-case'/fn,skiprows=4,index_col=0)
                    temp_prev_df = temp_prev_df.loc[new_index]
                    temp_prev_df.reset_index(inplace=True)
                    temp_prev_df.set_index('t',inplace=True)
                    prev_df = temp_prev_df.loc[prev_key_year]
                else:
                    prev_df = temp_df.loc[prev_key_year]
                next_df = temp_df.loc[next_key_year]
                prev_price = prev_df.loc[:,'total_cost_enduse'].values
                next_price = next_df.loc[:,'total_cost_enduse'].values
                prev_ghg = prev_df.loc[:,'aer_gen_co2e'].values
                next_ghg = next_df.loc[:,'aer_gen_co2e'].values
                prev_ng = cambium_ng.loc[prev_key_year].values
                next_ng = cambium_ng.loc[next_key_year].values
                prices = np.add(prev_price,np.multiply(interp_frac,np.subtract(next_price,prev_price)))
                ghgs = np.add(prev_ghg,np.multiply(interp_frac,np.subtract(next_ghg,prev_ghg)))
                ng = prev_ng+interp_frac*(next_ng-prev_ng)
                new_series = pd.Series(prices,new_index,name=year)
                cambium_prices = cambium_prices.join(new_series)
                new_series = pd.Series(ghgs,new_index,name=year)
                cambium_ghgs = cambium_ghgs.join(new_series)
                new_ng = pd.DataFrame(ng,index=[year],columns=['Val'])
                cambium_ng = pd.concat([cambium_ng,new_ng])
        cambium_prices = cambium_prices.sort_index(axis=1)
        cambium_ghgs = cambium_ghgs.sort_index(axis=1)
        cambium_ng = cambium_ng.sort_index()
        cambium_ng.loc[:] = inflate(cambium_ng.values, 2004, dollar_year)

        cambium_prices.to_json(resource_dir/('cambium_prices_'+name+'.json'))
        cambium_ghgs.to_json(resource_dir/('cambium_ghgs_'+name+'.json'))
        cambium_ng.to_json(resource_dir/('cambium_ng_'+name+'.json'))

def set_cambium_inputs(hi, cambium_scenario, year, state):

    resource_dir = Path(__file__).parent.absolute()/'inputs'

    with open(Path(resource_dir/('cambium_prices_'+cambium_scenario+'.json')),'r') as file:
        cambium_prices = pd.read_json(file)
    with open(Path(resource_dir/('cambium_ghgs_'+cambium_scenario+'.json')),'r') as file:
        cambium_ghgs = pd.read_json(file)
    with open(Path(resource_dir/('cambium_ng_'+cambium_scenario+'.json')),'r') as file:
        cambium_ng = pd.read_json(file)
    with open(Path(resource_dir/('cambium_aeo_multipliers.json')),'r') as file:
        state_multipliers = json.load(file)

    aeo_multiplier = state_multipliers[state]
    cambium_price = cambium_prices.loc[state,year:].values
    grid_co2e_kg_mwh = cambium_ghgs.loc[state,year:].values
    grid_wc_kg_mwh = all_grid_wcs[(year-2020):]

    cambium_price = np.mean(np.hstack([cambium_price,np.ones(31-len(list(cambium_price)))*cambium_price[-1]]))
    grid_co2e_kg_mwh = np.mean(np.hstack([grid_co2e_kg_mwh,np.ones(31-len(list(grid_co2e_kg_mwh)))*grid_co2e_kg_mwh[-1]]))
    grid_wc_kg_mwh = np.mean(np.hstack([grid_wc_kg_mwh,np.ones(31-len(list(grid_wc_kg_mwh)))*grid_wc_kg_mwh[-1]]))

    buy_price_kwh = cambium_price*aeo_multiplier/1000
    
    wind_lc = hi.system.wind._financial_model.lc_kwh
    wind_kwh = hi.system.wind.annual_energy_kwh
    pv_lc = hi.system.pv._financial_model.lc_kwh
    pv_kwh = hi.system.pv.annual_energy_kwh
    ppa_price_kwh = (wind_lc*wind_ppa_lcoe_ratio*wind_kwh+pv_lc*solar_ppa_lcoe_ratio*pv_kwh)/(wind_kwh+pv_kwh)

    setattr(hi.system.grid_purchase._financial_model,'voc_kwh',buy_price_kwh)
    setattr(hi.system.grid_sales._financial_model,'voc_kwh',ppa_price_kwh)
    hi.system.grid_purchase.config.lca['co2_kg_kwh'] = grid_co2e_kg_mwh/1000
    hi.system.grid_sales.config.lca['co2_kg_kwh'] = grid_co2e_kg_mwh/1000
    hi.system.grid_purchase.config.lca['h2o_kg_kwh'] = grid_wc_kg_mwh/1000
    hi.system.grid_sales.config.lca['h2o_kg_kwh'] = grid_wc_kg_mwh/1000

    ng_price_mmbtu = cambium_ng.loc[year].values[0]
    kJ_btu = 1.05506 # kJ/BTU
    NG_LHV_MJ_kg = 47.21 # Natural gas net calorific value, MJ/kg,
    ng_price_kg = ng_price_mmbtu/kJ_btu/1000*NG_LHV_MJ_kg
    setattr(hi.system.ng._financial_model,'voc_kg',ng_price_kg)

    return hi


if __name__ == '__main__':

    cambium_dir = Path(__file__).parent.absolute()/'inputs'/'Cambium data'
    extract_cambium_data(cambium_dir, 2020)