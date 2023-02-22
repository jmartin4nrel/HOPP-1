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

def run_hopp_calc(site, sim_tech, technologies, sim_cost, on_land):
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
        
    # Get interconnection size and pricing
    iconn_kw = technologies['interconnection']['capacity_kw']
    sell_price = technologies['interconnection']['ppa_sell_price_kwh']
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
    # TODO: modify financials from HOPP defaults
    
    # Simulate lifetime
    hybrid_plant.simulate_power()
    wind_kw = list(hybrid_plant.generation_profile.wind[0:8760])
    pv_kw = list(hybrid_plant.generation_profile.pv[0:8760])
    gen_kw = list(hybrid_plant.generation_profile.hybrid[0:8760])
    
    # Calculate lcoe before grid exchange for H2
    pv_toc_kwyr = sim_cost['pv']['total_annual_cost_kw']
    wind_toc_kwyr = sim_cost['wind']['total_annual_cost_kw']
    pv_size_kw = sim_tech['pv']['system_capacity_kw']
    num_turbines = sim_tech['wind']['num_turbines']
    turbine_rating_kw = sim_tech['wind']['turbine_rating_kw']                                   
    wind_size_kw = num_turbines*turbine_rating_kw
    toc_yr = pv_toc_kwyr*pv_size_kw + wind_toc_kwyr*wind_size_kw
    lcoe = toc_yr/sum(gen_kw)
    orig_lcoe = copy.copy(lcoe)
    orig_total_cost = copy.deepcopy(toc_yr)

     # plt.plot(np.arange(0,8760),gen_kw,label='Initial hybrid plant output')

    # Find excess generation above electrolyzer capcity and sell to grid
    profit_from_selling_to_grid = 0.0
    excess_energy = [0]*8760    
    for i in range(len(gen_kw)):
        if gen_kw[i] > elyzer_size_kw:
            excess_energy[i] = (gen_kw[i]-elyzer_size_kw)
            profit_from_selling_to_grid += (gen_kw[i]-elyzer_size_kw)*sell_price
            gen_kw[i] = elyzer_size_kw
    
    # plt.plot(np.arange(0,8760),gen_kw,label='After selling excess to grid')

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
    for i in range(len(gen_kw)):
        cost_to_buy_from_grid += purchase[i]*buy_price
        gen_kw[i] += purchase[i]

    new_total_cost = orig_total_cost+cost_to_buy_from_grid-profit_from_selling_to_grid
    lcoe = new_total_cost/sum(gen_kw)
    
    # plt.plot(np.arange(0,8760),gen_kw,label='After buying from grid for H2')
    # plt.xlabel('[hr]')
    # plt.ylabel('[kW]')
    # plt.legend()
    # plt.show()

    # Save outputs
    outputs = dict()
    outputs['kW from wind'] = np.sum(wind_kw)/8760
    outputs['kW from PV'] = np.sum(pv_kw)/8760
    outputs['Original LCOE [$/kWh]'] = orig_lcoe
    outputs['LCOE [$/kWh]'] = lcoe
    outputs['kW to H2 electrolysis'] = np.sum(gen_kw)/8760
    outputs['kW bought from grid'] = np.sum(purchase)/8760
    outputs['kW sold to grid'] = np.sum(excess_energy)/8760
    
    return outputs, site.wind_resource.filename, site.solar_resource.filename


def run_hybrid_calc_bruteforce(site_name, year, site_num, res_fn_wind, res_fn_solar, lat, lon, on_land,
                                technologies, costs, results_dir, plant_pct, wind_pct):
    
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
    
    # Make reduced version of technologies dict that has just what HOPP uses to setup technologies dict
    sim_tech = copy.deepcopy(technologies)
    sim_cost = copy.deepcopy(costs)
    if on_land == 'true':
        sim_tech['wind'] = technologies['lbw']
        sim_cost['wind'] = costs['lbw']
    else:
        sim_tech['wind'] = technologies['osw']
        sim_cost['wind'] = costs['osw']
    sim_tech.pop('lbw')
    sim_tech.pop('osw')
    sim_tech.pop('interconnection')
    sim_cost.pop('lbw')
    sim_cost.pop('osw')

    # Establish site location
    Site = sample_site  # sample_site has been loaded from flatirons_site to provide sample site boundary information
    Site['lat'] = lat
    Site['lon'] = lon
    Site['site_num'] = site_num
    Site['resource_filename_solar'] = res_fn_solar
    Site['resource_filename_wind'] = res_fn_wind
    Site['year'] = year

    # Get the Timezone offset value based on the lat/lon of the site
    try:
        location = {'lat': Site['lat'], 'long': Site['lon']}
        tz_val = get_offset(**location)
        Site['tz'] = (tz_val - 1)
    except:
        print('Timezone lookup failed for {}'.format(location))
    
    # Wait to de-synchronize api requests on multi-threaded analysis #TODO: get multiple api keys so this is not necessary
    wait = 10+rng.integers(10)
    time.sleep(wait)
    
    # Create site, downloading resource files if needed
    Site = SiteInfo(Site, hub_height=sim_tech['wind']['hub_height'],
                    solar_resource_file=Site['resource_filename_solar'],
                    wind_resource_file=Site['resource_filename_wind'])

    # Run HOPP calculation
    hopp_outputs, res_fn_wind, res_fn_solar = run_hopp_calc(Site, sim_tech, technologies, sim_cost, on_land)
    filename = '{}{:02d}_plant{:03d}_wind{:02d}.txt'.format(site_name,site_num,plant_pct,wind_pct)
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


def run_all_hybrid_calcs(site_name, site_details, technologies_lol, costs, results_dir,
                        plant_size_pcts, wind_pcts, optimize=False):

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
            all_arg = [site_name, site_details['year'][i],site_num,
                        site_details['wind_filenames'][i], site_details['solar_filenames'][i],
                        site_details['lat'][i], site_details['lon'][i], site_details['on_land'][i],
                        technologies_lol[j][k], costs, results_dir,
                        plant_pct, wind_pct]
            all_args.append(all_arg)
        else:
            for j, plant_pct in  enumerate(plant_size_pcts):
                for k, wind_pct in enumerate(wind_pcts):
                    all_arg = [site_name, site_details['year'][i],site_num,
                                site_details['wind_filenames'][i], site_details['solar_filenames'][i],
                                site_details['lat'][i], site_details['lon'][i], site_details['on_land'][i],
                                technologies_lol[j][k], costs, results_dir,
                                plant_pct, wind_pct]
                    all_args.append(all_arg)
        

    # # Run a multi-threaded analysis
    # with multiprocessing.Pool(9) as p:
    #     if optimize:
    #         p.starmap(run_hybrid_calc_optimize, all_args)
    #     else:
    #         p.starmap(run_hybrid_calc_bruteforce, all_args)

    # Run a single-threaded analysis
    for all_arg in all_args:
        if optimize:
            run_hybrid_calc_optimize(*all_args)
        else:
            run_hybrid_calc_bruteforce(*all_arg)


if __name__ == '__main__':

    # Set paths
    current_dir = Path(__file__).parent.absolute()
    resource_dir = current_dir/'..'/'..'/'resource_files'
    results_dir = current_dir/'..'/'resource_files'/'methanol_RCC'/'HOPP_results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    load_resource_from_file = True

    # Load dump files from data import
    with open(Path(results_dir/'..'/'engin.json'),'r') as file:
        engin = json.load(file)
    with open(Path(results_dir/'..'/'finance.json'),'r') as file:
        finance = json.load(file)
    with open(Path(results_dir/'..'/'scenario.json'),'r') as file:
        scenario_info = json.load(file)
        plant_scenarios = scenario_info['plant_scenarios']
        atb_scenarios = scenario_info['atb_scenarios']
        H2A_scenarios = scenario_info['H2A_scenarios']
        MeOH_scenarios = scenario_info['MeOH_scenarios']
    with open(Path(results_dir/'..'/'locations.json'),'r') as file:
        locations = json.load(file)

    # Set Analysis Location and Details
    resource_year = 2013
    sim_years = scenario_info['sim_years']
    plant_size_pcts = [60]#np.arange(60,150,10)
    wind_pcts = [40]#np.arange(10,100,10)
    site_name_list = list(locations.keys())[:1]
    sites_per_location = 1
    
    for site_name in site_name_list:
        
        desired_lats = locations[site_name]['lat'][:sites_per_location]
        desired_lons = locations[site_name]['lon'][:sites_per_location]
        
        locations[site_name]['lcoe_$_kwh'] = [[]*len(desired_lats)]
        locations[site_name]['pv_capacity_kw'] = [[]*len(desired_lats)]
        locations[site_name]['wind_capacity_kw'] = [[]*len(desired_lats)]
        locations[site_name]['pv_output_kw'] = [[]*len(desired_lats)]
        locations[site_name]['wind_output_kw'] = [[]*len(desired_lats)]

        for year_idx, sim_year in enumerate(sim_years[3:]):
        
            # Load wind and solar resource files for location nearest desired lats and lons
            # NB this resource information will be overriden by API retrieved data if load_resource_from_file is set to False
            sitelist_name = 'filtered_site_details_{}_locs_{}_year'.format(len(desired_lats), resource_year)
            # sitelist_name = 'site_details.csv'
            if load_resource_from_file:
                # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
                site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, resource_year, not_rect=True,\
                                                    max_dist=.01)  # Return contains
                site_details.insert(3,'on_land',locations[site_name]['on_land'][:1])
                # site_details = filter_sites(site_details, location='usa only')
                site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))
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

            site_nums = site_details['site_nums'][:sites_per_location]

            # Constants needed by HOPP
            solar_tracking_mode = '1-axis'
            ppa_sell_price_kwh = 0.01 #TODO setup variable ppa
            ppa_buy_price_kwh = 0.05 #TODO setup variable ppa
            correct_wind_speed_for_height = True
            
            # Get H2 elyzer size #TODO: Add year lookup
            H2A_scenario = plant_scenarios['H2']
            elyzer_input_kw = engin['H2']['elec_in_kw'][H2A_scenario][year_idx]
            elyzer_cf = 0.97 #TODO: variable electrolyzer capacity
            elyzer_size_kw = elyzer_input_kw/elyzer_cf

            # Estimate wind/solar needed based on capacity factor #TODO: Add year lookup
            pv_scenario = plant_scenarios['PV']
            pv_cap = engin['PV']['capacity_factor'][pv_scenario][year_idx]
            lbw_scenario = plant_scenarios['LBW']
            lbw_cap = engin['LBW']['capacity_factor'][lbw_scenario][year_idx]
            osw_scenario = plant_scenarios['OSW']
            osw_cap = engin['OSW']['capacity_factor'][osw_scenario][year_idx]

            if site_details['on_land'][0] == 'false':
                wind_pcts = [100]
            technologies_lol = []
            
            for i, plant_pct in enumerate(plant_size_pcts):
                technologies_list = []   
                costs_list = []   
                for j, wind_pct in enumerate(wind_pcts):

                    osw_input_kw = elyzer_input_kw*plant_pct/100
                    lbw_input_kw = elyzer_input_kw*plant_pct/100*wind_pct/100
                    pv_input_kw= elyzer_input_kw*plant_pct/100*(100-wind_pct)/100
                    
                    osw_size_kw = osw_input_kw/osw_cap
                    lbw_size_kw = lbw_input_kw/lbw_cap
                    pv_size_kw = pv_input_kw/pv_cap

                    iconn_kw = [osw_size_kw,pv_size_kw+lbw_size_kw]

                    lbw_turb_rating_kw = engin['LBW']['turbine_rating_kw'][lbw_scenario][year_idx]
                    lbw_hub_height = engin['LBW']['hub_height'][lbw_scenario][-1]
                    lbw_rotor_diameter = engin['LBW']['rotor_diameter'][lbw_scenario][year_idx]

                    osw_turb_rating_kw = engin['OSW']['turbine_rating_kw'][osw_scenario][year_idx]
                    osw_hub_height = engin['OSW']['hub_height'][osw_scenario][-1]
                    osw_rotor_diameter = engin['OSW']['rotor_diameter'][osw_scenario][year_idx]

                    technologies = {'pv': {
                                        'system_capacity_kw': pv_size_kw
                                    },
                                    'lbw': {
                                        'num_turbines': round(lbw_size_kw/lbw_turb_rating_kw),
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
                                        'ppa_sell_price_kwh': ppa_sell_price_kwh
                                    }
                                    }

                    technologies_list.append(technologies)

                technologies_lol.append(technologies_list)

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
            year_results_dir = results_dir/str(sim_year)
            if not os.path.exists(year_results_dir):
                os.mkdir(year_results_dir)
            if not os.path.exists(year_results_dir/'OrigLCOE'):
                os.mkdir(year_results_dir/'OrigLCOE')
            if not os.path.exists(year_results_dir/'LCOE'):
                os.mkdir(year_results_dir/'LCOE')
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

            # Run hybrid calculation for all sites
            tic = time.time()
            run_all_hybrid_calcs(site_name, site_details, technologies_lol, costs,
                                    year_results_dir, plant_size_pcts, wind_pcts)
            toc = time.time()
            print('Time to complete 1 set of calcs: {:.2f} min'.format((toc-tic)/60))
            
            for site_num in site_nums:
            
                min_lcoe = np.inf
                opt_pv = np.inf
                opt_wind = np.inf

                for i, plant_pct in enumerate(plant_size_pcts):
                    if locations[site_name]['on_land'][0] == 'false':
                        final_wind_pcts = [100]
                    else:
                        final_wind_pcts = wind_pcts
                    for j, wind_pct in enumerate(final_wind_pcts):
                        fn = '{}{:02d}_plant{:03d}_wind{:02d}.txt'.format(site_name,site_num,plant_pct,wind_pct)
                        new_lcoe = float(np.loadtxt(year_results_dir/'LCOE'/fn))
                        if new_lcoe < min_lcoe:
                            min_lcoe = copy.copy(new_lcoe)
                            
                            osw_input_kw = elyzer_input_kw*plant_pct/100
                            lbw_input_kw = elyzer_input_kw*plant_pct/100*wind_pct/100
                            pv_input_kw= elyzer_input_kw*plant_pct/100*(100-wind_pct)/100
                            
                            osw_size_kw = osw_input_kw/osw_cap
                            lbw_size_kw = lbw_input_kw/lbw_cap
                            pv_size_kw = pv_input_kw/pv_cap

                            pv_output = float(np.loadtxt(year_results_dir/'kWPV'/fn))
                            wind_output = float(np.loadtxt(year_results_dir/'kWwind'/fn))

                            if locations[site_name]['on_land'][0] == 'false':
                                opt_pv = 0
                                opt_wind = copy.copy(osw_size_kw)
                            else:
                                opt_pv = copy.copy(pv_size_kw)
                                opt_wind = copy.copy(lbw_size_kw)

                locations[site_name]['lcoe_$_kwh'][site_num-1].append(min_lcoe)
                locations[site_name]['pv_capacity_kw'][site_num-1].append(opt_pv)
                locations[site_name]['pv_output_kw'][site_num-1].append(pv_output)
                locations[site_name]['wind_capacity_kw'][site_num-1].append(opt_wind)
                locations[site_name]['wind_output_kw'][site_num-1].append(wind_output)


    resource_dir = current_dir/'..'/'resource_files'/'methanol_RCC'
    with open(Path(resource_dir/'locations.json'),'w') as file:
        out_locations = copy.deepcopy(locations)
        for ID, loc in out_locations.items():
            for i, value in enumerate(loc['on_land']):
                value = str(value).lower()
                loc['on_land'][i] = value
        json.dump(out_locations, file)

