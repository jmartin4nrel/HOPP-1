import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def inflate(dollars, original_year, new_year):

    CPI = [172.2, 177.1, 179.9, 184.0, 188.9, 195.3, 201.6, 207.3, 215.3, 214.5,
           218.1, 224.9, 229.6, 233.0, 236.7, 237.0, 240.0, 245.1, 251.1, 255.7,
           258.8, 271.0, 292.7] # US City Average, all items, https://data.bls.gov/pdq/SurveyOutputServlet
    orig_cpi = CPI[int(original_year)-2000]
    new_cpi = CPI[int(new_year)-2000]
    dollars *= new_cpi/orig_cpi

    # standard_inflation = 0.025
    # dollars *= (1+standard_inflation)**(new_year-original_year)

    return dollars
    

cambium_dir = Path(__file__).parent.absolute()/'..'/'..'/'..'/'..'/'..'/'Projects'/'22 CO2 to Methanol'/'Cambium Data'

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
    prices = inflate(temp_price.values, (year-2020)/2+2019, 2021)
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

# Add cambium prices for different scenarios
key_years = [2022,2024,2026,2028,2030,2035,2040,2045,2050]
years = np.arange(2023,2051)
cambium_prices = {}
cambium_ghgs = {}
cambium_ng = {}
for scenario, name in scenario_names.items():
    cambium_prices[scenario] = copy.deepcopy(annual_cambium_prices)
    cambium_ghgs[scenario] = copy.deepcopy(annual_cambium_ghgs)
    cambium_ng[scenario] = pd.read_csv(cambium_dir/scenario/'repgasprice_nat.csv',index_col=0)
    cambium_ng[scenario] = cambium_ng[scenario].loc[2020:2050]
    ng2021 = pd.DataFrame(np.mean(cambium_ng[scenario].loc[2020:2022].values),index=[2021],columns=['Val'])
    cambium_ng[scenario] = pd.concat([cambium_ng[scenario],ng2021])
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
        new_series = pd.Series(prices,new_index,name=year)
        cambium_prices[scenario] = cambium_prices[scenario].join(new_series)
        new_series = pd.Series(temp_ghg.values,new_index,name=year)
        cambium_ghgs[scenario] = cambium_ghgs[scenario].join(new_series)
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
            prev_ng = cambium_ng[scenario].loc[prev_key_year].values
            next_ng = cambium_ng[scenario].loc[next_key_year].values
            prices = np.add(prev_price,np.multiply(interp_frac,np.subtract(next_price,prev_price)))
            ghgs = np.add(prev_ghg,np.multiply(interp_frac,np.subtract(next_ghg,prev_ghg)))
            ng = prev_ng+interp_frac*(next_ng-prev_ng)
            new_series = pd.Series(prices,new_index,name=year)
            cambium_prices[scenario] = cambium_prices[scenario].join(new_series)
            new_series = pd.Series(ghgs,new_index,name=year)
            cambium_ghgs[scenario] = cambium_ghgs[scenario].join(new_series)
            new_ng = pd.DataFrame(ng,index=[year],columns=['Val'])
            cambium_ng[scenario] = pd.concat([cambium_ng[scenario],new_ng])
    cambium_prices[scenario] = cambium_prices[scenario].sort_index(axis=1)
    cambium_ghgs[scenario] = cambium_ghgs[scenario].sort_index(axis=1)
    cambium_ng[scenario] = cambium_ng[scenario].sort_index()
    cambium_ng[scenario].loc[:] = inflate(cambium_ng[scenario].values, 2004, 2021)