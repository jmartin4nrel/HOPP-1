"""
multi_location_RCC.py

Simulates hybrid plants over a targeted region of the US for H2 electrolysis
(& eventual MeOH production w/NGCC plant captured CO2)
"""

from email.base64mime import header_length
import os
import sys
import copy
import json
import pandas as pd
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import multiprocessing
import operator
from pathlib import Path
from itertools import repeat
import time
rng = np.random.default_rng()

import matplotlib.pyplot as plt

from hybrid.keys import set_nrel_key_dot_env
from hybrid.log import analysis_logger as logger
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from tools.analysis import create_cost_calculator
from tools.resource import *
from tools.resource.resource_loader import site_details_creator

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", None, "display.max_columns", None)

set_nrel_key_dot_env()

def run_hopp_calc(site, sim_tech, technologies, sim_cost, on_land, sim_power, just_WC=False, prev_results_dir=None, prev_filename=None):
    """ run_hopp_calc Establishes sizing models, creates a wind or solar farm based on the desired sizes,
     and runs SAM model calculations for the specified inputs.
     save_outputs contains a dictionary of all results for the hopp calculation.

    :param site: Hybrid plant site set up with location info
    :param sim_tech: dict of solar/wind technology for setting up HybridSimulation
    :param technologies: dict of additional tech info
    :param on_land: boolean, whether site is on land
    :param lifetime: int, plant lifetime in years
    :return: dict with LCOE and amount of electricity used for electrolysis, plus amount bought and sold,
     plus wind and solar filenames used
    """
    
    if just_WC:

        # Load original carbon intensities
        hybrid_CI = technologies['interconnection']['hybrid_co2e_kg_mwh']
        grid_CI = list(technologies['interconnection']['grid_co2e_kg_mwh'])
        plant_lifespan = 30
        if len(grid_CI)<plant_lifespan:
            grid_CI.extend([grid_CI[-1]]*(plant_lifespan-len(grid_CI)))
        grid_CI = np.mean(grid_CI)
        Orig_CI_fp = prev_results_dir/'OrigCI'/prev_filename
        CI_fp = prev_results_dir/'CI'/prev_filename
        Orig_CI = np.loadtxt(Orig_CI_fp)
        CI = np.loadtxt(CI_fp)
        Orig_CI = float(Orig_CI)
        CI = float(CI)

        # Find percent CI change from grid exchange
        grid_hybrid_diff = grid_CI-hybrid_CI
        new_orig_diff = CI-Orig_CI
        change_ratio = new_orig_diff/grid_hybrid_diff

        # Calculate water consumption ratios
        hybrid_WC = technologies['interconnection']['hybrid_wc_kg_mwh']
        grid_WC = list(technologies['interconnection']['grid_wc_kg_mwh'])
        if len(grid_WC)<plant_lifespan:
            grid_WC.extend([grid_WC[-1]]*(plant_lifespan-len(grid_WC)))
        grid_WC = np.mean(grid_WC)
        grid_hybrid_diff_WC = (grid_WC-hybrid_WC)
        new_orig_diff = grid_hybrid_diff_WC*change_ratio
        final_WC = hybrid_WC+new_orig_diff

        outputs = dict()
        outputs['Original WC [g H2O/kWh]'] = hybrid_WC
        outputs['WC [g H2O/kWh]'] = final_WC

        return outputs, site.wind_resource.filename, site.solar_resource.filename

    else:

        # TODO: make plant lifespan input
        plant_lifespan = 30

        # Get interconnection size and pricing
        iconn_kw = technologies['interconnection']['capacity_kw']
        buy_price = technologies['interconnection']['ppa_buy_price_kwh']

        # Determine interconnection size based on whether on land or offshore
        if on_land == 'true':
            iconn_size_kw = iconn_kw[1]
        else:
            iconn_size_kw = iconn_kw[0]

        # Get electrolysis needs
        elyzer_size_kw = technologies['pem']['capacity_kw']
        elyzer_cf = technologies['pem']['capacity_factor']

        # Set up plant
        hybrid_plant = HybridSimulation(sim_tech, site, iconn_size_kw)
        
        # Modify pvwatts model to reflect ATB technology
        pv_factors = sim_power['PV']
        for factor, value in pv_factors.items():
            # Some factors have to be iterables even if constant - dumb!
            if factor in ['albedo','dc_degradation']:
                value = (value,)
            hybrid_plant.pv._system_model.value(factor,value)
        

        # Modify windpower model to reflect ATB technology
        wind_factors = sim_power['wind']
        for factor, value in wind_factors.items():
            hybrid_plant.wind._system_model.value(factor,value)
        
        # # Check wind power curve
        # x = hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds
        # y = hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout
        # plt.clf()
        # plt.plot(x,y)
        # plt.show()

        # Simulate lifetime
        hybrid_plant.simulate_power()
        wind_kw = list(hybrid_plant.generation_profile.wind[0:8760])
        pv_kw = list(hybrid_plant.generation_profile.pv[0:8760])
        gen_kw = list(hybrid_plant.generation_profile.hybrid[0:8760])
        
        # # Check solar power curve
        # pv_poa = list(hybrid_plant.pv._system_model.Outputs.poa[0:8760])
        # plt.clf()
        # plt.plot(pv_poa,pv_kw,'.')
        # plt.show()
                    
        # Calculate lcoe before grid exchange for H2
        pv_toc_kwyr = sim_cost['pv']['total_annual_cost_kw']
        wind_toc_kwyr = sim_cost['wind']['total_annual_cost_kw']
        pv_size_kw = sim_tech['pv']['system_capacity_kw']/sim_power['PV']['dc_ac_ratio']
        num_turbines = sim_tech['wind']['num_turbines']
        turbine_rating_kw = sim_tech['wind']['turbine_rating_kw']                                   
        wind_size_kw = num_turbines*turbine_rating_kw
        toc_yr = pv_toc_kwyr*pv_size_kw + wind_toc_kwyr*wind_size_kw
        lcoe = toc_yr/sum(gen_kw)
        orig_lcoe = copy.copy(lcoe)
        orig_total_cost = copy.deepcopy(toc_yr)
        
        # Calculate carbon intensity/water consumption before grid exchange
        hybrid_CI = technologies['interconnection']['hybrid_co2e_kg_mwh']
        grid_CI = technologies['interconnection']['grid_co2e_kg_mwh']
        hybrid_WC = technologies['interconnection']['hybrid_wc_kg_mwh']
        grid_WC = technologies['interconnection']['grid_wc_kg_mwh']
        ghg_kg = [i/1000*hybrid_CI for i in gen_kw]
        wc_kg = [i/1000*hybrid_WC for i in gen_kw]
        orig_ghg_kg = copy.deepcopy(ghg_kg)
        orig_wc_kg = copy.deepcopy(wc_kg)
        orig_gen_kw = copy.deepcopy(gen_kw)
        orig_CI = np.sum(orig_ghg_kg)/np.sum(gen_kw)*1000
        orig_WC = np.sum(orig_wc_kg)/np.sum(gen_kw)*1000

        # plt.plot(np.arange(0,8760),[i/1000 for i in gen_kw],label='Wind/solar above 144 MW - difference sold under PPA')

        # Find excess generation above electrolyzer capcity and sell to grid
        ppa_lcoe_ratio = technologies['interconnection']['ppa_lcoe_ratio']
        sell_price = orig_lcoe*ppa_lcoe_ratio
        profit_from_selling_to_grid = 0.0
        excess_energy = [0]*8760    
        for i in range(len(gen_kw)):
            if gen_kw[i] > elyzer_size_kw:
                excess_energy[i] = (gen_kw[i]-elyzer_size_kw)
                profit_from_selling_to_grid += (gen_kw[i]-elyzer_size_kw)*sell_price
                gen_kw[i] = elyzer_size_kw
        
        # plt.plot(np.arange(0,8760),[i/1000 for i in gen_kw],label='Wind/solar under 136 MW - difference purchased from grid')

        # Buy grid electricity to meet electrolyzer capacity factor
        cost_to_buy_from_grid = 0.0
        purchase_needed = (elyzer_size_kw*elyzer_cf-np.mean(gen_kw))*8760
        shortfall = np.subtract(elyzer_size_kw,gen_kw)
        shortfall_inds = np.flip(np.argsort(shortfall))
        diff_shortfall = -np.diff(shortfall[shortfall_inds])
        shortfall_changes = np.squeeze(np.argwhere(diff_shortfall))
        purchase = np.zeros(8760)
        shortfall_change = 1
        while np.sum(purchase) < purchase_needed and shortfall_change < len(shortfall_changes):
            purchase[shortfall_inds[:(1+shortfall_changes[shortfall_change])]] += diff_shortfall[shortfall_changes[shortfall_change-1]]
            shortfall_change += 1
        extra_purchase = sum(purchase)-purchase_needed
        avg_extra_purchase = extra_purchase/(1+shortfall_changes[shortfall_change-1])
        purchase[shortfall_inds[:(1+shortfall_changes[shortfall_change])]] -= avg_extra_purchase
        for i in range(len(gen_kw)):
            gen_kw[i] += purchase[i]
        for year_idx in range(min(len(buy_price),plant_lifespan)):
            year_buy_price = buy_price[year_idx]
            for i in range(len(gen_kw)):
                cost_to_buy_from_grid += purchase[i]*year_buy_price
        for year_idx in range(plant_lifespan-len(buy_price)):
            year_buy_price = buy_price[min(len(buy_price),plant_lifespan)-1]
            for i in range(len(gen_kw)):
                cost_to_buy_from_grid += purchase[i]*year_buy_price
        cost_to_buy_from_grid /= plant_lifespan

        new_total_cost = orig_total_cost+cost_to_buy_from_grid-profit_from_selling_to_grid
        lcoe = new_total_cost/sum(gen_kw)

        # Find change in CI and WC from net grid exchange
        ghg_change_kg = [0]*8760 
        wc_change_kg = [0]*8760 
        net_exchange_kw = np.subtract(purchase,excess_energy)
        for year_idx in range(min(len(buy_price),plant_lifespan)):
            exchange_CI_kg_kwh = (grid_CI[year_idx]-hybrid_CI)/1000
            exchange_WC_kg_kwh = (grid_WC[year_idx]-hybrid_WC)/1000
            ghg_change_kg = np.add(ghg_change_kg,np.multiply(exchange_CI_kg_kwh,net_exchange_kw))
            wc_change_kg = np.add(wc_change_kg,np.multiply(exchange_WC_kg_kwh,net_exchange_kw))
        for year_idx in range(plant_lifespan-len(buy_price)):
            exchange_CI_kg_kwh = (grid_CI[min(len(buy_price),plant_lifespan)-1]-hybrid_CI)/1000
            exchange_WC_kg_kwh = (grid_WC[min(len(buy_price),plant_lifespan)-1]-hybrid_WC)/1000
            ghg_change_kg = np.add(ghg_change_kg,np.multiply(exchange_CI_kg_kwh,net_exchange_kw))
            wc_change_kg = np.add(wc_change_kg,np.multiply(exchange_WC_kg_kwh,net_exchange_kw))
        ghg_change_kg = [i/plant_lifespan for i in ghg_change_kg]
        wc_change_kg = [i/plant_lifespan for i in wc_change_kg]
        final_ghg_kg = np.add(orig_ghg_kg,ghg_change_kg)
        final_wc_kg = np.add(orig_wc_kg,wc_change_kg)
        final_CI = np.sum(final_ghg_kg)/np.sum(orig_gen_kw)*1000
        final_WC = np.sum(final_wc_kg)/np.sum(orig_gen_kw)*1000
        
        # plt.plot(np.arange(0,8760),[i/1000 for i in gen_kw],label='Electrolyzer maintained between 136 MW and 144 MW')
        # plt.xlabel('[hr of year]')
        # plt.ylabel('[MW]')
        # plt.legend()
        # plt.xlim([0,500])
        # plt.ylim([0,420])
        # plt.show()

        # Save outputs
        outputs = dict()
        outputs['kW from wind'] = np.sum(wind_kw)/8760
        outputs['kW from PV'] = np.sum(pv_kw)/8760
        outputs['Original LCOE [$/kWh]'] = orig_lcoe
        outputs['LCOE [$/kWh]'] = lcoe
        outputs['Original CI [g CO2e/kWh]'] = orig_CI
        outputs['CI [g CO2e/kWh]'] = final_CI
        outputs['Original WC [g H2O/kWh]'] = orig_WC
        outputs['WC [g H2O/kWh]'] = final_WC
        outputs['kW to H2 electrolysis'] = np.sum(gen_kw)/8760
        outputs['kW bought from grid'] = np.sum(purchase)/8760
        outputs['kW sold to grid'] = np.sum(excess_energy)/8760
        
        return outputs, site.wind_resource.filename, site.solar_resource.filename


def run_hybrid_calc_bruteforce(site_name, year, site_num, res_fn_wind, res_fn_solar, lat, lon, on_land,
                                technologies, costs, results_dir, plant_pct, wind_pct, power_factors):
    
    """
    run_hybrid_calc loads the specified resource for each site, and runs wind, solar, hybrid and solar addition
    scenarios by calling run_hopp_calc for each scenario. Returns a DataFrame of all results for the supplied site
    :param year: year for which analysis is conducted
    :param site_num: number representing the site studied. Generally 1 to num sites to be studied.
    :param res_fn_wind: filename of wind resource file.
    :param res_fn_solar: filename of solar resource file.
    :param lat: site latitude (degrees).
    :param lon: site longitude (degrees).
    :param on_land: boolean, whether site is on land
    :param technologies: dict of technologies setup info to send to hybrid simulation setup.
    :param results_dir: path to results directory.
    :return: save_outputs_resource_loop_dataframe <pandas dataframe> dataframe of all site outputs from hopp runs
    """

    plant = technologies['interconnection']['H2_plant']
    filename = '{}{:02d}_plant{:03d}_wind{:02d}_{}.txt'.format(site_name,site_num,plant_pct,wind_pct,plant)
    if True: #filename not in os.listdir(results_dir/'OrigLCOE'):

        # Make reduced version of technologies dict that has just what HOPP uses to setup technologies dict
        sim_tech = copy.deepcopy(technologies)
        sim_cost = copy.deepcopy(costs)
        sim_power = copy.deepcopy(power_factors)
        if on_land == 'true':
            sim_tech['wind'] = technologies['lbw']
            sim_cost['wind'] = costs['lbw']
            sim_power['wind'] = power_factors['LBW']
        else:
            sim_tech['wind'] = technologies['osw']
            sim_cost['wind'] = costs['osw']
            sim_power['wind'] = power_factors['OSW']
        sim_tech.pop('lbw')
        sim_tech.pop('osw')
        sim_tech.pop('interconnection')
        sim_cost.pop('lbw')
        sim_cost.pop('osw')
        sim_power.pop('LBW')
        sim_power.pop('OSW')

        # Establish site location
        Site = {}
        Site['lat'] = lat
        Site['lon'] = lon
        Site['year'] = year
        # For site area: square with sides = sqrt of number of turbines times rotor diameter times ten
        d = sim_tech['wind']['rotor_diameter']
        n = sim_tech['wind']['num_turbines']
        side = 10*d*n**.5
        Site['site_boundaries'] = {'verts':[[0,0],[0,side],[side,side],[side,0]]}

        # Get the Timezone offset value based on the lat/lon of the site
        try:
            location = {'lat': Site['lat'], 'long': Site['lon']}
            tz_val = get_offset(**location)
            Site['tz'] = (tz_val - 1)
        except:
            print('Timezone lookup failed for {}'.format(location))
        
        # # Wait to de-synchronize api requests on multi-threaded analysis #TODO: get multiple api keys so this is not necessary
        # wait = 10+rng.integers(10)
        # time.sleep(wait)
        
        # Create site, downloading resource files if needed
        Site = SiteInfo(Site, hub_height=sim_tech['wind']['hub_height'],
                        solar_resource_file=res_fn_solar,
                        wind_resource_file=res_fn_wind)

        # Get ppa lcoe_ratio
        wind_ppa_lcoe_ratio = technologies['interconnection']['wind_ppa_lcoe_ratio']
        solar_ppa_lcoe_ratio = technologies['interconnection']['solar_ppa_lcoe_ratio']
        ppa_lcoe_ratio = (wind_ppa_lcoe_ratio*wind_pct+solar_ppa_lcoe_ratio*(100-wind_pct))/100
        technologies['interconnection']['ppa_lcoe_ratio'] = ppa_lcoe_ratio

        # Get ghgs/water consumption
        wind_ghgs = 29.49369802
        solar_ghgs = 22.38638471
        wind_wcs = 0.429207
        solar_wcs = -0.02128 
        ghgs = (wind_ghgs*wind_pct+solar_ghgs*(100-wind_pct))/100
        wcs = (wind_wcs*wind_pct+solar_wcs*(100-wind_pct))/100
        technologies['interconnection']['hybrid_co2e_kg_mwh'] = ghgs
        technologies['interconnection']['hybrid_wc_kg_mwh'] = wcs

        # Run HOPP calculation
        hopp_outputs, res_fn_wind, res_fn_solar = run_hopp_calc(Site, sim_tech, technologies, sim_cost, on_land, sim_power)
        print('Finished site '+filename)

        # Write resulst to text files #TODO make big .json
        results_filepath = results_dir/'OrigLCOE'/filename
        np.savetxt(results_filepath,[hopp_outputs['Original LCOE [$/kWh]']])
        results_filepath = results_dir/'LCOE'/filename
        np.savetxt(results_filepath,[hopp_outputs['LCOE [$/kWh]']])
        results_filepath = results_dir/'kWH2'/filename
        np.savetxt(results_filepath,[hopp_outputs['kW to H2 electrolysis']])
        results_filepath = results_dir/'kWwind'/filename
        np.savetxt(results_filepath,[hopp_outputs['kW from wind']])
        results_filepath = results_dir/'kWPV'/filename
        np.savetxt(results_filepath,[hopp_outputs['kW from PV']])
        results_filepath = results_dir/'kWbuy'/filename
        np.savetxt(results_filepath,[hopp_outputs['kW bought from grid']])
        results_filepath = results_dir/'kWsell'/filename
        np.savetxt(results_filepath,[hopp_outputs['kW sold to grid']])
        results_filepath = results_dir/'OrigCI'/filename
        np.savetxt(results_filepath,[hopp_outputs['Original CI [g CO2e/kWh]']])
        results_filepath = results_dir/'CI'/filename
        np.savetxt(results_filepath,[hopp_outputs['CI [g CO2e/kWh]']])
        results_filepath = results_dir/'OrigWC'/filename
        np.savetxt(results_filepath,[hopp_outputs['Original WC [g H2O/kWh]']])
        results_filepath = results_dir/'WC'/filename
        np.savetxt(results_filepath,[hopp_outputs['WC [g H2O/kWh]']])

    elif filename not in os.listdir(results_dir/'OrigWC'):

        # Make reduced version of technologies dict that has just what HOPP uses to setup technologies dict
        sim_tech = copy.deepcopy(technologies)
        sim_cost = copy.deepcopy(costs)
        sim_power = copy.deepcopy(power_factors)
        if on_land == 'true':
            sim_tech['wind'] = technologies['lbw']
            sim_cost['wind'] = costs['lbw']
            sim_power['wind'] = power_factors['LBW']
        else:
            sim_tech['wind'] = technologies['osw']
            sim_cost['wind'] = costs['osw']
            sim_power['wind'] = power_factors['OSW']
        sim_tech.pop('lbw')
        sim_tech.pop('osw')
        sim_tech.pop('interconnection')
        sim_cost.pop('lbw')
        sim_cost.pop('osw')
        sim_power.pop('LBW')
        sim_power.pop('OSW')

        # Establish site location
        Site = {}
        Site['lat'] = lat
        Site['lon'] = lon
        Site['year'] = year
        # For site area: square with sides = sqrt of number of turbines times rotor diameter times ten
        d = sim_tech['wind']['rotor_diameter']
        n = sim_tech['wind']['num_turbines']
        side = 10*d*n**.5
        Site['site_boundaries'] = {'verts':[[0,0],[0,side],[side,side],[side,0]]}

        # Get the Timezone offset value based on the lat/lon of the site
        try:
            location = {'lat': Site['lat'], 'long': Site['lon']}
            tz_val = get_offset(**location)
            Site['tz'] = (tz_val - 1)
        except:
            print('Timezone lookup failed for {}'.format(location))
        
        # # Wait to de-synchronize api requests on multi-threaded analysis #TODO: get multiple api keys so this is not necessary
        # wait = 10+rng.integers(10)
        # time.sleep(wait)
        
        # Create site, downloading resource files if needed
        Site = SiteInfo(Site, hub_height=sim_tech['wind']['hub_height'],
                        solar_resource_file=res_fn_solar,
                        wind_resource_file=res_fn_wind)

        # Get ppa lcoe_ratio
        wind_ppa_lcoe_ratio = technologies['interconnection']['wind_ppa_lcoe_ratio']
        solar_ppa_lcoe_ratio = technologies['interconnection']['solar_ppa_lcoe_ratio']
        ppa_lcoe_ratio = (wind_ppa_lcoe_ratio*wind_pct+solar_ppa_lcoe_ratio*(100-wind_pct))/100
        technologies['interconnection']['ppa_lcoe_ratio'] = ppa_lcoe_ratio

        # Get ghgs/water consumption
        wind_ghgs = 29.49369802
        solar_ghgs = 22.38638471
        wind_wcs = 0.429207
        solar_wcs = -0.02128 
        ghgs = (wind_ghgs*wind_pct+solar_ghgs*(100-wind_pct))/100
        wcs = (wind_wcs*wind_pct+solar_wcs*(100-wind_pct))/100
        technologies['interconnection']['hybrid_co2e_kg_mwh'] = ghgs
        technologies['interconnection']['hybrid_wc_kg_mwh'] = wcs

        # Run HOPP calculation
        hopp_outputs, res_fn_wind, res_fn_solar = run_hopp_calc(Site, sim_tech, technologies, sim_cost, on_land, sim_power, just_WC=True, prev_results_dir=results_dir, prev_filename=filename)
        print('Finished site '+filename)

        # Write resulst to text files #TODO make big .json
        results_filepath = results_dir/'OrigWC'/filename
        np.savetxt(results_filepath,[hopp_outputs['Original WC [g H2O/kWh]']])
        results_filepath = results_dir/'WC'/filename
        np.savetxt(results_filepath,[hopp_outputs['WC [g H2O/kWh]']])

def run_all_hybrid_calcs(site_name, site_details, technologies_lols, costs, results_dir,
                        plant_size_pcts, wind_pcts, power_factors, optimize=False):

    """
    Performs a multi-threaded run of run_hybrid_calc for the given input parameters.
    Returns a dataframe result for all sites
    :param site_details: DataFrame containing site details for all sites to be analyzed,
    including site_nums, lat, long, wind resource filename and solar resource filename.
    :param technologies: dict of technologies setup info to send to hybrid simulation setup.
    :param results_dir: path to results directory
    :param other_attrs: other attributes of the system, such as financials
    :return: DataFrame of results for run_hybrid_calc at all sites (save_all_runs)
    """

    all_args = []

    for i, site_num in enumerate(site_details['site_nums']):
        if optimize:
            j = int(np.ceil(len(plant_size_pcts)/2))
            plant_pct = plant_size_pcts[j]
            k = int(np.ceil(len(wind_pcts)/2))
            wind_pct = wind_pcts[k]
            for l in range(len(technologies_lols[j][k])):
                all_arg = [site_name, site_details['year'][i],site_num,
                            site_details['wind_filenames'][i], site_details['solar_filenames'][i],
                            site_details['lat'][i], site_details['lon'][i], site_details['on_land'][i],
                            technologies_lols[j][k][l], costs, results_dir,
                            plant_pct, wind_pct, power_factors]
                all_args.append(all_arg)
        else:
            for j, plant_pct in  enumerate(plant_size_pcts):
                for k, wind_pct in enumerate(wind_pcts):
                    for l in range(len(technologies_lols[j][k])):
                        all_arg = [site_name, site_details['year'][i],site_num,
                                    site_details['wind_filenames'][i], site_details['solar_filenames'][i],
                                    site_details['lat'][i], site_details['lon'][i], site_details['on_land'][i],
                                    technologies_lols[j][k][l], costs, results_dir,
                                    plant_pct, wind_pct, power_factors]
                        all_args.append(all_arg)
        

    # # Run a multi-threaded analysis
    with multiprocessing.Pool(10) as p:
        if optimize:
            p.starmap(run_hybrid_calc_optimize, all_args)
        else:
            p.starmap(run_hybrid_calc_bruteforce, all_args)

    # # Run a single-threaded analysis
    # for all_arg in all_args:
    #     if optimize:
    #         run_hybrid_calc_optimize(*all_args)
    #     else:
    #         run_hybrid_calc_bruteforce(*all_arg)


if __name__ == '__main__':

    # Set paths
    current_dir = Path(__file__).parent.absolute()
    resource_dir = current_dir/'..'/'resource_files'/'methanol_RCC'
    load_resource_from_file = False

    # Load dump files from data import
    cambium_scenarios = ['MidCase']#,'HighNGPrice','LowNGPrice'
    for l, cambium_scenario in enumerate(cambium_scenarios):   
        resource_dir = current_dir/'..'/'resource_files'/'methanol_RCC'
        results_dir = current_dir/'..'/'resource_files'/'methanol_RCC'/'HOPP_results'/cambium_scenario
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
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
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

        # Set Analysis Location and Details
        resource_year = 2013
        sim_years = scenario_info['sim_years']
        plant_size_pcts = np.arange(100,200,20)
        wind_pcts = np.arange(90,110,20)
        site_name_list = list(locations.keys())#[1:2]
        site_name_list.reverse()
        sites_per_location = 19
        single_site = True

        if single_site:
            site_name_list = [scenario_info['site_selection']['site_name']]
            site_nums = [scenario_info['site_selection']['site_num']]
            sites_per_location = 1
        else:
            site_nums = np.arange(1,sites_per_location+1,1)

        resource_dir = current_dir/'..'/'..'/'resource_files'

        for site_name in site_name_list:
            
            if single_site:
                desired_lats = [locations[site_name]['lat'][site_nums[0]-1]]
                desired_lons = [locations[site_name]['lon'][site_nums[0]-1]]
            else:
                desired_lats = locations[site_name]['lat'][:sites_per_location]
                desired_lons = locations[site_name]['lon'][:sites_per_location]

            for plant in ['HCO2','HPSR']:#
            
                locations[site_name][plant] = {}
                locations[site_name][plant]['orig_lcoe_$_kwh'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['lcoe_$_kwh'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['orig_CI_g_kwh'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['CI_g_kwh'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['orig_WC_g_kwh'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['WC_g_kwh'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['pv_capacity_kw'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['wind_capacity_kw'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['pv_output_kw'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['wind_output_kw'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['electrolyzer_input_kw'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['grid_bought_kw'] = [[] for i in range(len(desired_lats))]
                locations[site_name][plant]['grid_sold_kw'] = [[] for i in range(len(desired_lats))]

            for sim_year in sim_years:

                year_idx = scenario_info['sim_years'].index(sim_year)
            
                # Load wind and solar resource files for location nearest desired lats and lons
                # NB this resource information will be overriden by API retrieved data if load_resource_from_file is set to False
                sitelist_name = 'filtered_site_details_{}_locs_{}_year'.format(len(desired_lats), resource_year)
                # sitelist_name = 'site_details.csv'
                if load_resource_from_file:
                    # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
                    site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, resource_year, not_rect=True,\
                                                        max_dist=.01)  # Return contains
                    if single_site:
                        site_details.insert(3,'on_land',locations[site_name]['on_land'][site_nums[0]])
                        site_details['site_nums'] = site_nums
                    else:
                        site_details.insert(3,'on_land',locations[site_name]['on_land'][:sites_per_location])
                    # site_details = filter_sites(site_details, location='usa only')
                    site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))#.format(site_name)))
                    # print(site_name,len(np.unique(site_details['solar_filenames']))-np.sum([i == '' for i in site_details['solar_filenames']]))
                else:
                    # Creates the site_details file containing grid of lats, lons, years, and wind and solar filenames (blank
                    # - to force API resource download)
                    if os.path.exists(sitelist_name):
                        site_details = pd.read_csv(sitelist_name)
                    else:
                        site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, resource_year, not_rect=True)
                        # Filter to locations in USA - MOVE TO PARALLEL
                        site_details = filter_sites(site_details, location='usa only')
                        site_details.to_csv(sitelist_name)

                # site_nums = site_details['site_nums'][:sites_per_location]

                # Constants needed by HOPP
                correct_wind_speed_for_height = True
                
                # Get H2 elyzer size
                H2_plants = ['HCO2','HPSR']#
                HCO2_scenario = plant_scenarios['HCO2']
                HPSR_scenario = plant_scenarios['HPSR']
                
                elyzer_inputs_kw = [engin['HCO2']['elec_in_kw'][HCO2_scenario][year_idx],engin['HPSR']['elec_in_kw'][HPSR_scenario][year_idx]]#
                elyzer_cf = 0.97 #TODO: variable electrolyzer capacity
                elyzer_sizes_kw = [i/elyzer_cf for i in elyzer_inputs_kw]

                # Get grid pricing & ghgs
                wind_ppa_lcoe_ratio = 0.7742
                solar_ppa_lcoe_ratio = 0.6959
                cambium_aeo_multiplier = state_multipliers[site_name[:2]]
                cambium_price = cambium_prices.loc[site_name[:2],sim_year:].values
                grid_co2e_kg_mwh = cambium_ghgs.loc[site_name[:2],sim_year:].values
                all_grid_wcs = [3237.82422	,
                                3046.291736	,
                                2854.759253	,
                                2663.226769	,
                                2471.694286	,
                                2280.161802	,
                                2312.016492	,
                                2343.871183	,
                                2375.725873	,
                                2407.580564	,
                                2439.435254	,
                                2350.330575	,
                                2261.225896	,
                                2172.121216	,
                                2083.016537	,
                                1993.911858	,
                                1981.843038	,
                                1969.774217	,
                                1957.705397	,
                                1945.636576	,
                                1933.567756	,
                                1932.402861	,
                                1931.237967	,
                                1930.073072	,
                                1928.908178	,
                                1927.743283	,
                                1913.587052	,
                                1899.430821	,
                                1885.27459	,
                                1871.118359	,
                                1856.962128]
                all_grid_wcs = [1718.605915	,
                                1576.131286	,
                                1433.656658	,
                                1291.182029	,
                                1148.707401	,
                                1006.232772	,
                                877.3949737	,
                                748.5571751	,
                                619.7193766	,
                                490.881578	,
                                362.0437794	,
                                331.9507828	,
                                301.8577862	,
                                271.7647896	,
                                241.671793	,
                                211.5787964	,
                                204.5211937	,
                                197.463591	,
                                190.4059883	,
                                183.3483856	,
                                176.2907829	,
                                175.0760655	,
                                173.861348	,
                                172.6466306	,
                                171.4319132	,
                                170.2171957	,
                                167.6368848	,
                                165.0565739	,
                                162.476263	,
                                159.8959522	,
                                157.3156413]
                grid_wc_kg_mwh = all_grid_wcs[(sim_year-2020):]
                ppa_buy_price_kwh = cambium_price*cambium_aeo_multiplier/1000
                
                # Estimate wind/solar needed based on capacity factor
                pv_scenario = plant_scenarios['PV']
                pv_cap = engin['PV']['capacity_factor'][pv_scenario][year_idx]
                lbw_scenario = plant_scenarios['LBW']
                lbw_cap = engin['LBW']['capacity_factor'][lbw_scenario][year_idx]
                osw_scenario = plant_scenarios['OSW']
                osw_cap = engin['OSW']['capacity_factor'][osw_scenario][year_idx]

                # Make dict of power factors
                power_factors_to_set = {
                    'PV':['albedo','array_type','azimuth','bifaciality','dc_ac_ratio',
                        'dc_degradation','inv_eff','losses'],
                    'LBW':['wind_turbine_max_cp','avail_bop_loss','avail_grid_loss',
                        'avail_turb_loss','elec_eff_loss','elec_parasitic_loss',
                        'env_degrad_loss','env_env_loss','env_icing_loss','ops_env_loss',
                        'ops_grid_loss','ops_load_loss','turb_generic_loss',
                        'turb_hysteresis_loss','turb_perf_loss','turb_specific_loss',
                        'wake_ext_loss']
                }
                power_factors_to_set['OSW'] = power_factors_to_set['LBW']
                power_factors = {}
                for tech, factor_list in power_factors_to_set.items():
                    power_factors[tech] = {}
                    for factor in factor_list:
                        scenario = scenario_info['plant_scenarios'][tech]
                        power_factor = engin[tech][factor][scenario]
                        if type(power_factor) is list:
                            power_factor = power_factor[year_idx]
                        power_factors[tech][factor] = power_factor
                        

                if site_details['on_land'][0] == 'false':
                    wind_pcts = [100]
                technologies_lols = []
                
                for plant_pct in plant_size_pcts:
                    technologies_lol = []   
                    costs_list = []   
                    for wind_pct in wind_pcts:
                        technologies_list = []
                        for i, elyzer_input_kw in enumerate(elyzer_inputs_kw):
                            elyzer_size_kw = elyzer_sizes_kw[i]

                            osw_input_kw = elyzer_input_kw*plant_pct/100
                            lbw_input_kw = elyzer_input_kw*plant_pct/100*wind_pct/100
                            pv_input_kw= elyzer_input_kw*plant_pct/100*(100-wind_pct)/100
                            
                            osw_size_kw = osw_input_kw/osw_cap
                            lbw_size_kw = lbw_input_kw/lbw_cap
                            pv_size_kw = pv_input_kw/pv_cap
                            pv_dc_size_kw = pv_size_kw*power_factors['PV']['dc_ac_ratio']

                            lbw_turb_rating_kw = engin['LBW']['turbine_rating_kw'][lbw_scenario][year_idx]
                            lbw_hub_height = engin['LBW']['hub_height'][lbw_scenario][year_idx]
                            lbw_rotor_diameter = engin['LBW']['rotor_diameter'][lbw_scenario][year_idx]
                            num_turbines = round(lbw_size_kw/lbw_turb_rating_kw)
                            if num_turbines > 300:
                                num_turbines = 300
                                lbw_size_kw = num_turbines*lbw_turb_rating_kw

                            iconn_kw = [osw_size_kw,pv_size_kw+lbw_size_kw]

                            osw_turb_rating_kw = engin['OSW']['turbine_rating_kw'][osw_scenario][year_idx]
                            osw_hub_height = engin['OSW']['hub_height'][osw_scenario][year_idx]
                            osw_rotor_diameter = engin['OSW']['rotor_diameter'][osw_scenario][year_idx]

                            technologies = {'pv': {
                                                'system_capacity_kw': pv_dc_size_kw,
                                            },
                                            'lbw': {
                                                'num_turbines': num_turbines,
                                                'turbine_rating_kw': lbw_turb_rating_kw,
                                                'hub_height': lbw_hub_height,
                                                'rotor_diameter': lbw_rotor_diameter
                                            },
                                            'osw': {
                                                'num_turbines': round(osw_size_kw/osw_turb_rating_kw),
                                                'turbine_rating_kw': osw_turb_rating_kw,
                                                'hub_height': osw_hub_height,
                                                'rotor_diameter': osw_rotor_diameter
                                            },
                                            'pem': {
                                                'capacity_kw': elyzer_size_kw,
                                                'capacity_factor': elyzer_cf,
                                            },
                                            'interconnection': {
                                                'capacity_kw': iconn_kw,
                                                'ppa_buy_price_kwh': ppa_buy_price_kwh,
                                                'wind_ppa_lcoe_ratio': wind_ppa_lcoe_ratio,
                                                'solar_ppa_lcoe_ratio': solar_ppa_lcoe_ratio,
                                                'grid_co2e_kg_mwh': grid_co2e_kg_mwh,
                                                'grid_wc_kg_mwh': grid_wc_kg_mwh,
                                                'H2_plant': H2_plants[i]
                                            }
                                            }

                            technologies_list.append(technologies)

                        technologies_lol.append(technologies_list)

                    technologies_lols.append(technologies_lol)

                sim_basis_year = finance['PV']['basis_year']
                plant_lifespan = finance['PV']['plant_lifespan']
                discount_rate = finance['PV']['discount_rate']
                TASC_multiplier = finance['PV']['TASC_multiplier']
                
                pv_occ_kw = finance['PV']['OCC_$_kw'][pv_scenario][year_idx]
                pv_occ_kwyr = pv_occ_kw*TASC_multiplier*discount_rate
                pv_fom_kwyr = finance['PV']['FOM_$_kwyr'][pv_scenario][year_idx]
                pv_toc_kwyr = pv_occ_kwyr + pv_fom_kwyr
                
                lbw_occ_kw = finance['LBW']['OCC_$_kw'][lbw_scenario][year_idx]
                lbw_occ_kwyr = lbw_occ_kw*TASC_multiplier*discount_rate
                lbw_fom_kwyr = finance['LBW']['FOM_$_kwyr'][lbw_scenario][year_idx]
                lbw_toc_kwyr = lbw_occ_kwyr + lbw_fom_kwyr

                osw_occ_kw = finance['OSW']['OCC_$_kw'][osw_scenario][year_idx]
                osw_occ_kwyr = osw_occ_kw*TASC_multiplier*discount_rate
                osw_fom_kwyr = finance['OSW']['FOM_$_kwyr'][osw_scenario][year_idx]
                osw_toc_kwyr = osw_occ_kwyr + osw_fom_kwyr

                costs ={'pv': {
                            'total_annual_cost_kw': pv_toc_kwyr
                        },
                        'lbw': {
                            'total_annual_cost_kw': lbw_toc_kwyr
                        },
                        'osw': {
                            'total_annual_cost_kw': osw_toc_kwyr
                        }}   # TODO :Add other inputs such as pricing
                


                # Save results from all locations to folder
                year_results_dir = results_dir/(str(sim_year))# +' geo'
                if not os.path.exists(year_results_dir):
                    os.mkdir(year_results_dir)
                if not os.path.exists(year_results_dir/'OrigLCOE'):
                    os.mkdir(year_results_dir/'OrigLCOE')
                if not os.path.exists(year_results_dir/'LCOE'):
                    os.mkdir(year_results_dir/'LCOE')
                if not os.path.exists(year_results_dir/'OrigCI'):
                    os.mkdir(year_results_dir/'OrigCI')
                if not os.path.exists(year_results_dir/'CI'):
                    os.mkdir(year_results_dir/'CI')
                if not os.path.exists(year_results_dir/'OrigWC'):
                    os.mkdir(year_results_dir/'OrigWC')
                if not os.path.exists(year_results_dir/'WC'):
                    os.mkdir(year_results_dir/'WC')
                if not os.path.exists(year_results_dir/'kWH2'):
                    os.mkdir(year_results_dir/'kWH2')
                if not os.path.exists(year_results_dir/'kWwind'):
                    os.mkdir(year_results_dir/'kWwind')
                if not os.path.exists(year_results_dir/'kWPV'):
                    os.mkdir(year_results_dir/'kWPV')
                if not os.path.exists(year_results_dir/'kWbuy'):
                    os.mkdir(year_results_dir/'kWbuy')
                if not os.path.exists(year_results_dir/'kWsell'):
                    os.mkdir(year_results_dir/'kWsell')

                # # Run hybrid calculation for all sites
                # tic = time.time()
                # run_all_hybrid_calcs(site_name, site_details, technologies_lols, costs,
                #                         year_results_dir, plant_size_pcts, wind_pcts, power_factors)
                # toc = time.time()
                # print('Time to complete 1 set of calcs: {:.2f} min'.format((toc-tic)/60))
                
                for site_num in site_nums:
                
                    for k, plant in enumerate(['HCO2','HPSR']):#
                        
                        min_lcoe = np.inf
                        min_CI = np.inf
                        opt_pv = np.inf
                        opt_wind = np.inf

                        elyzer_input_kw = elyzer_inputs_kw[k]

                        for i, plant_pct in enumerate(plant_size_pcts):
                            if locations[site_name]['on_land'][0] == 'false':
                                final_wind_pcts = [100]
                            else:
                                final_wind_pcts = wind_pcts
                            for j, wind_pct in enumerate(final_wind_pcts):
                                fn = '{}{:02d}_plant{:03d}_wind{:02d}_{}.txt'.format(site_name,site_num,plant_pct,wind_pct,plant)
                                new_lcoe = float(np.loadtxt(year_results_dir/'LCOE'/fn))
                                new_CI = float(np.loadtxt(year_results_dir/'CI'/fn))
                                if new_lcoe < min_lcoe:
                                    min_lcoe = copy.copy(new_lcoe)
                                    min_CI = copy.copy(new_CI)
                                    
                                    osw_input_kw = elyzer_input_kw*plant_pct/100
                                    lbw_input_kw = elyzer_input_kw*plant_pct/100*wind_pct/100
                                    pv_input_kw= elyzer_input_kw*plant_pct/100*(100-wind_pct)/100
                                    
                                    osw_size_kw = osw_input_kw/osw_cap
                                    lbw_size_kw = lbw_input_kw/lbw_cap
                                    pv_size_kw = pv_input_kw/pv_cap

                                    num_turbines = round(lbw_size_kw/lbw_turb_rating_kw)
                                    lbw_size_kw = num_turbines*lbw_turb_rating_kw

                                    num_osw_turbines = round(osw_size_kw/osw_turb_rating_kw)
                                    osw_size_kw = num_osw_turbines*osw_turb_rating_kw

                                    orig_lcoe = float(np.loadtxt(year_results_dir/'OrigLCOE'/fn))
                                    orig_CI = float(np.loadtxt(year_results_dir/'OrigCI'/fn))
                                    CI = float(np.loadtxt(year_results_dir/'CI'/fn))
                                    orig_WC = float(np.loadtxt(year_results_dir/'OrigWC'/fn))
                                    WC = float(np.loadtxt(year_results_dir/'WC'/fn))
                                    pv_output = float(np.loadtxt(year_results_dir/'kWPV'/fn))
                                    wind_output = float(np.loadtxt(year_results_dir/'kWwind'/fn))
                                    elec_input = float(np.loadtxt(year_results_dir/'kWH2'/fn))
                                    grid_bought = float(np.loadtxt(year_results_dir/'kWbuy'/fn))
                                    grid_sold = float(np.loadtxt(year_results_dir/'kWsell'/fn))

                                    if locations[site_name]['on_land'][0] == 'false':
                                        opt_pv = 0
                                        opt_wind = copy.copy(osw_size_kw)
                                    else:
                                        opt_pv = copy.copy(pv_size_kw)
                                        opt_wind = copy.copy(lbw_size_kw)

                        if single_site:
                            locations[site_name][plant]['lcoe_$_kwh'][0].append(min_lcoe)
                            locations[site_name][plant]['orig_lcoe_$_kwh'][0].append(orig_lcoe)
                            locations[site_name][plant]['CI_g_kwh'][0].append(CI)
                            locations[site_name][plant]['orig_CI_g_kwh'][0].append(orig_CI)
                            locations[site_name][plant]['WC_g_kwh'][0].append(WC)
                            locations[site_name][plant]['orig_WC_g_kwh'][0].append(orig_WC)
                            locations[site_name][plant]['pv_capacity_kw'][0].append(opt_pv)
                            locations[site_name][plant]['pv_output_kw'][0].append(pv_output)
                            locations[site_name][plant]['wind_capacity_kw'][0].append(opt_wind)
                            locations[site_name][plant]['wind_output_kw'][0].append(wind_output)
                            locations[site_name][plant]['electrolyzer_input_kw'][0].append(elec_input)
                            locations[site_name][plant]['grid_bought_kw'][0].append(grid_bought)
                            locations[site_name][plant]['grid_sold_kw'][0].append(grid_sold)
                        else:
                            locations[site_name][plant]['lcoe_$_kwh'][site_num-1].append(min_lcoe)
                            locations[site_name][plant]['orig_lcoe_$_kwh'][site_num-1].append(orig_lcoe)
                            locations[site_name][plant]['CI_g_kwh'][site_num-1].append(CI)
                            locations[site_name][plant]['orig_CI_g_kwh'][site_num-1].append(orig_CI)
                            locations[site_name][plant]['WC_g_kwh'][site_num-1].append(WC)
                            locations[site_name][plant]['orig_WC_g_kwh'][site_num-1].append(orig_WC)
                            locations[site_name][plant]['pv_capacity_kw'][site_num-1].append(opt_pv)
                            locations[site_name][plant]['pv_output_kw'][site_num-1].append(pv_output)
                            locations[site_name][plant]['wind_capacity_kw'][site_num-1].append(opt_wind)
                            locations[site_name][plant]['wind_output_kw'][site_num-1].append(wind_output)
                            locations[site_name][plant]['electrolyzer_input_kw'][site_num-1].append(elec_input)
                            locations[site_name][plant]['grid_bought_kw'][site_num-1].append(grid_bought)
                            locations[site_name][plant]['grid_sold_kw'][site_num-1].append(grid_sold)


        resource_dir = current_dir/'..'/'resource_files'/'methanol_RCC'/'HOPP_results'/cambium_scenario
        with open(Path(resource_dir/'locations.json'),'w') as file:
            out_locations = copy.deepcopy(locations)
            for ID, loc in out_locations.items():
                for i, value in enumerate(loc['on_land']):
                    value = str(value).lower()
                    loc['on_land'][i] = value
            json.dump(out_locations, file)

