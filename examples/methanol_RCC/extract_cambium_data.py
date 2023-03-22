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

scenarios = {'Mid-case':'MidCase',
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


annual_items = {'Energy cost [2020$/MWh]':'energy_cost_enduse',
              'Capacity cost [2020$/MWh]':'capacity_cost_enduse',
              'Portfolio cost [2020$/MWh]':'portfolio_cost_enduse'}

hourly_items = {'Natural Gas Price [2020$/MMBtu]':'repgasprice_nat.csv',
              'Energy cost [2020$/MWh]':'energy_cost_enduse',
              'Capacity cost [2020$/MWh]':'capacity_cost_enduse',
              'Portfolio cost [2020$/MWh]':'portfolio_cost_enduse',
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
annual_cambium_prices = {}
for key, name in annual_items.items():
    annual_cambium_prices[key] = pd.DataFrame(None,index=new_index)
    for year in [2020,2022]:
        fn = prefix[year]+'_'+scenarios['Mid-case']+'_annual_state.csv'
        temp_df = pd.read_csv(cambium_dir/'Mid-case'/fn,skiprows=year-2018,index_col=0)
        temp_df = temp_df.loc[new_index]
        temp_df.reset_index(inplace=True)
        temp_df.set_index('t',inplace=True)
        temp_df = temp_df.loc[year]
        temp_df = temp_df.loc[:,name]
        prices = inflate(temp_df.values, (year-2020)/2+2019, 2021)
        new_series = pd.Series(prices,new_index,name=year)
        annual_cambium_prices[key] = annual_cambium_prices[key].join(new_series)
    mid_prices = []
    for state in new_index:
        values = annual_cambium_prices[key].loc[state].values
        mid_prices.append(np.mean(values))
    new_series = pd.Series(mid_prices,new_index,name=2021)
    annual_cambium_prices[key] = annual_cambium_prices[key].join(new_series)
    annual_cambium_prices[key] = annual_cambium_prices[key].sort_index(axis=1)
    annual_cambium_prices[key] = annual_cambium_prices[key].sort_index()

# Correlate cambium data with AEO retail prices
A1 = []
A2 = []
A3 = []
b = []
for year in [2022,]:
    A1.extend(list(annual_cambium_prices['Energy cost [2020$/MWh]'].loc[:,year].values))
    A2.extend(list(annual_cambium_prices['Capacity cost [2020$/MWh]'].loc[:,year].values))
    A3.extend(list(annual_cambium_prices['Portfolio cost [2020$/MWh]'].loc[:,year].values))
    b.extend(list(aeo_prices.loc[:,year].values))
A_alt = [A1[i]+A2[i]+A3[i] for i in range(len(b))]
# A3 = [i*100+100 for i in A3]

# plt.plot(A_alt,b,'.')
# plt.show()

A_cols = [A_alt,A3]
# A_cols = [A1,A2,A3,np.ones(len(A_alt))]
A = np.hstack([np.transpose(np.array([i])) for i in A_cols])
A_means = np.mean(A, axis=0)
A_n = np.divide(A,A_means)
b_mean = np.mean(b)
b_n = np.divide(b,b_mean)
results = np.linalg.lstsq(A_n.astype(float),b_n.astype(float), rcond=None)
x = results[0]
alt_results = np.linalg.lstsq(A_n[:,[0,]].astype(float),b_n.astype(float), rcond=None)
x_alt = alt_results[0]

b_fit_alt = np.multiply(np.sum(np.multiply(A_n[:,[0,]],x_alt),axis=1),b_mean)
x_adj = np.multiply(np.divide(x,A_means),b_mean)
b_fit = np.sum(np.multiply(A,x_adj),axis=1)

plt.subplot(1,2,1)
plt.plot(b_fit,b,'.')
plt.subplot(1,2,2)
plt.plot(b_fit_alt,b,'.')
# plt.show()

plt.clf()
ax = plt.figure().add_subplot(projection='3d')
ax.plot(A_alt,A3,b,'.')
plt.show()

years = np.arange(2020,2051)

