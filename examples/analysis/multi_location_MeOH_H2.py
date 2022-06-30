"""
multi_location_MeOH_H2.py

Simulates ~833 MW hybrid plants over a targeted region of the US for H2 electrolysis
(& eventual MeOH production w/NGCC plant captured CO2)
"""

from email.base64mime import header_length
import os
import sys
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

from hybrid.keys import set_nrel_key_dot_env
from hybrid.log import analysis_logger as logger
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from tools.analysis import create_cost_calculator
from tools.resource import *
from tools.resource.resource_loader import site_details_creator

from examples.H2_Analysis.run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM

import warnings
warnings.filterwarnings("ignore")

resource_dir = Path(__file__).parent.parent.parent / "resource_files"

pd.set_option("display.max_rows", None, "display.max_columns", None)

set_nrel_key_dot_env()


def run_hopp_calc(Site, scenario_description, bos_details, total_hybrid_plant_capacity_mw, solar_size_mw, wind_size_mw,
                    nameplate_mw, electrolyzer_size_mw, load_resource_from_file,
                    ppa_price, results_dir):
    """ run_hopp_calc Establishes sizing models, creates a wind or solar farm based on the desired sizes,
     and runs SAM model calculations for the specified inputs.
     save_outputs contains a dictionary of all results for the hopp calculation.

    :param scenario_description: Project scenario - 'greenfield' or 'solar addition'.
    :param bos_details: contains bos details including type of analysis to conduct (cost/mw, json lookup, HybridBOSSE).
    :param total_hybrid_plant_capacity_mw: capacity in MW of hybrid plant.
    :param solar_size_mw: capacity in MW of solar component of plant.
    :param wind_size_mw: capacity in MW of wind component of plant.
    :param nameplate_mw: nameplate capacity of total plant.
    :param electrolyzer_size_mw: electrolyzer size in MW.
    :param load_resource_from_file: flag determining whether resource is loaded directly from file or through
     interpolation routine.
    :param ppa_price: PPA price in USD($)
    :return: collection of outputs from SAM and hybrid-specific calculations (includes e.g. AEP, IRR, LCOE),
     plus wind and solar filenames used
    (save_outputs)
    """
    # Get resource data
    if load_resource_from_file:
        pass
    else:
        Site['resource_filename_solar'] = ""  # Unsetting resource filename to force API download of wind resource
        Site['resource_filename_wind'] = ""  # Unsetting resource filename to force API download of solar resource

    # print('Establishing site number {}'.format(Site['site_num']))
    
    # Set up technology and cost model info
    turb_rating_kw = 7000
    tower_height = 110
    turbine_rating_mw = 7
    rotor_diameter = 200
    storage_size_mwh = 1
    storage_size_mw = 1
    storage_used = True
    battery_can_grid_charge = False
    run_reopt_flag = False
    critical_load_factor = 1
    storage_hours = storage_size_mwh/storage_size_mw
    wind_cost_kw = 671
    solar_cost_kw = 598
    storage_cost_kw = 97
    storage_cost_kwh = 104
    electrolyzer_cost_kw = 100
    custom_powercurve = True
    kw_continuous = electrolyzer_size_mw*1000
    load = [kw_continuous for x in
                        range(0, 8760)]

    
    site = SiteInfo(Site, hub_height=tower_height,
                    solar_resource_file=Site['resource_filename_solar'],
                    wind_resource_file=Site['resource_filename_wind'])

    num_turbines = int(wind_size_mw * 1000 / turb_rating_kw)
    technologies = {'pv': {
                        'system_capacity_kw': solar_size_mw * 1000
                    },          # mw system capacity
                    'wind': {
                        'num_turbines': num_turbines,
                        'turbine_rating_kw': turb_rating_kw,
                        'hub_height': tower_height,
                        'rotor_diameter': rotor_diameter
                    },
                    'battery': {
                        'system_capacity_kwh': storage_size_mwh * 1000,
                        'system_capacity_kw': storage_size_mw * 1000
                        }
                    }

    # Set up scenario
    scenario = dict()
    scenario['Scenario Number'] = 0
    scenario['Scenario Name'] = '2030 Advanced'
    scenario['Site Name'] = 'Name'
    scenario['Electrolyzer Size MW'] = electrolyzer_size_mw
    scenario['Wind Size MW'] = wind_size_mw
    scenario['Solar Size MW'] = solar_size_mw
    scenario['Storage Size MW'] = storage_size_mw
    scenario['Storage Size MWh'] = storage_size_mwh
    scenario['Lat'] = Site['lat']
    scenario['Long'] = Site['lon']
    scenario['ATB Year'] = 2030
    scenario['Powercurve File'] = 'powercurve_2020_atb_advanced'
    scenario['PTC Available'] = 'yes'
    scenario['ITC Available'] = 'no'
    scenario['Debt Equity'] = 60
    scenario['Discount Rate'] = 0.07
    scenario['Turbine Rating'] = turbine_rating_mw
    scenario['Tower Height'] = tower_height
    scenario['Rotor Diameter'] = rotor_diameter
    scenario['Wind Cost KW'] = wind_cost_kw
    scenario['Solar Cost KW'] = solar_cost_kw
    scenario['Storage Cost KW'] = storage_cost_kw
    scenario['Storage Cost KWh'] = storage_cost_kwh
    scenario['Electrolyzer Cost KW'] = electrolyzer_cost_kw
    scenario['Buy From Grid ($/kWh)'] = False
    scenario['Sell To Grid ($/kWh)'] = False
    scenario['Useful Life'] = 30
    useful_life = scenario['Useful Life']
    

    # Dummy ReOPT run to setup needed input for Parangat model
    dummy_wind_size_mw, dummy_solar_size_mw, dummy_storage_size_mw,\
    dummy_storage_size_mwh, dummy_storage_hours, reopt_results, REoptResultsDF = run_reopt(site, scenario, load,
                                                electrolyzer_size_mw*1000,
                                                critical_load_factor, useful_life,
                  battery_can_grid_charge, storage_used, run_reopt_flag)

    # Create model
    hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
    energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
        hopp_for_h2(site, scenario, technologies,
                    wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
        wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
        kw_continuous, load,
        custom_powercurve,
        electrolyzer_size_mw, grid_connected_hopp=True)

    wind_installed_cost = hybrid_plant.wind.total_installed_cost
    solar_installed_cost = hybrid_plant.pv.total_installed_cost
    hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

    # Run Simple Dispatch Model

    bat_model = SimpleDispatch()
    bat_model.Nt = len(energy_shortfall_hopp)
    bat_model.curtailment = combined_pv_wind_curtailment_hopp
    bat_model.shortfall = energy_shortfall_hopp

    bat_model.battery_storage = storage_size_mwh * 1000
    bat_model.charge_rate = storage_size_mw * 1000
    bat_model.discharge_rate = storage_size_mw * 1000

    battery_used, excess_energy, battery_SOC = bat_model.run()
    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used

    sell_price = 0.01
    buy_price = 0.05

    if sell_price:
        profit_from_selling_to_grid = np.sum(excess_energy)*sell_price
    else:
        profit_from_selling_to_grid = 0.0

    if buy_price:
        cost_to_buy_from_grid = 0.0

        for i in range(len(combined_pv_wind_storage_power_production_hopp)):
            if combined_pv_wind_storage_power_production_hopp[i] < kw_continuous:
                cost_to_buy_from_grid += (kw_continuous-combined_pv_wind_storage_power_production_hopp[i])*buy_price
                combined_pv_wind_storage_power_production_hopp[i] = kw_continuous
    else:
        cost_to_buy_from_grid = 0.0

    energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production_hopp]

    # Run the Python H2A model
                
    # electrical_generation_timeseries = combined_pv_wind_storage_power_production_hopp
    electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
    electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

    # Parangat model
    adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
    #NB: adjusted_installed_cost does NOT include the electrolyzer cost
    
    net_capital_costs = reopt_results['outputs']['Scenario']['Site'] \
                        ['Financial']['net_capital_costs']

    # intalled costs:
    # hybrid_plant.grid._financial_model.costs

    # system_rating = electrolyzer_size
    forced_electrolyzer_cost = electrolyzer_cost_kw
    system_rating = wind_size_mw + solar_size_mw
    H2_Results, H2A_Results = run_h2_PEM.run_h2_PEM(electrical_generation_timeseries,electrolyzer_size_mw,
                    kw_continuous,forced_electrolyzer_cost,lcoe,adjusted_installed_cost,useful_life,
                    net_capital_costs)

    # TEMPORARY CORRECTION FOR PEM EFFICIENCY.
    # # Convert H2 production from ~72.55kWh eff to 55.5kWh/kg
    H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output'] * 72.55/55.5
    
    # Intermediate financial calculation
    total_elec_production = np.sum(electrical_generation_timeseries) #REMOVE
    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
    total_electrolyzer_cost = H2A_Results['scaled_total_installed_cost']
    total_system_installed_cost = total_hopp_installed_cost + total_electrolyzer_cost
    annual_operating_cost_hopp = (wind_size_mw * 1000 * 42) + (solar_size_mw * 1000 * 13)
    annual_operating_cost_h2 = H2A_Results['Fixed O&M'] * H2_Results['hydrogen_annual_output']
    total_annual_operating_costs = annual_operating_cost_hopp + annual_operating_cost_h2 + cost_to_buy_from_grid - profit_from_selling_to_grid
    
    h_lcoe = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
            total_annual_operating_costs, 0.07, useful_life)

    # Save outputs
    outputs = dict()
    outputs['LCOH [$/kg]'] = h_lcoe
    outputs['H2 Production [tonne/day]'] = H2_Results['hydrogen_annual_output']/1000/365
    outputs['Total Generation [MWe]'] = np.sum(hybrid_plant.grid.generation_profile[0:8759])/1000/8760
    outputs['Sold to Grid [MWe]'] = np.sum(excess_energy)/1000/8760
    
    return outputs, site.wind_resource.filename, site.solar_resource.filename


def run_hybrid_calc(year, site_num, scenario_descriptions, results_dir, load_resource_from_file,
                    resource_filename_wind, resource_filename_solar, site_lat, site_lon,
                    wind_size, solar_size, hybrid_size, interconnection_size,
                    bos_details, ppa_price, solar_tracking_mode, hub_height, correct_wind_speed_for_height, all_run_dir):
    """
    run_hybrid_calc loads the specified resource for each site, and runs wind, solar, hybrid and solar addition
    scenarios by calling run_hopp_calc for each scenario. Returns a DataFrame of all results for the supplied site
    :param year: year for which analysis is conducted
    :param site_num: number representing the site studied. Generally 1 to num sites to be studied.
    :param scenario_descriptions: description of scenario type, e.g. wind only, solar only, hybrid.
    :param results_dir: path to results directory.
    :param load_resource_from_file: flag determining whether resource is loaded from file directly or other routine.
    :param resource_filename_wind: filename of wind resource file.
    :param resource_filename_solar: filename of solar resource file.
    :param site_lat: site latitude (degrees).
    :param site_lon: site longitude (degrees).
    :param wind_size: capacity in MW of wind component of plant.
    :param solar_size: capacity in MW of solar component of plant.
    :param hybrid_size: capacity in MW of hybrid plant.
    :param interconnection_size: capacity in MW of interconnection.
    :param bos_details: contains bos details including type of analysis to conduct (cost/mw,
     json lookup, HybridBOSSE).
    :param ppa_price: PPA price in USD($).
    :param solar_tracking_mode: solar tracking mode (e.g. fixed, single-axis, two-axis).
    :param hub_height: hub height in meters.
    :param correct_wind_speed_for_height: (boolean) flag determining whether wind speed is extrapolated
     to hub height.
    :param all_run_dir: path to directory for results from all runs
    :return: save_outputs_resource_loop_dataframe <pandas dataframe> dataframe of all site outputs from hopp runs
    """
    
    # Wait to de-synchronize api requests
    wait = 10+rng.integers(10)
    time.sleep(wait)

    # Set up hopp_outputs dictionary
    hopp_outputs = dict()

    # Site details
    Site = sample_site  # sample_site has been loaded from flatirons_site to provide sample site boundary information
    Site['lat'] = site_lat
    Site['lon'] = site_lon
    Site['site_num'] = site_num
    Site['resource_filename_solar'] = resource_filename_solar
    Site['resource_filename_wind'] = resource_filename_wind
    Site['year'] = year

    # Get the Timezone offset value based on the lat/lon of the site
    try:
        location = {'lat': Site['lat'], 'long': Site['lon']}
        tz_val = get_offset(**location)
        Site['tz'] = (tz_val - 1)
    except:
        print('Timezone lookup failed for {}'.format(location))

    scenario_description = 'greenfield'
    solar_size_mw = solar_size
    wind_size_mw = wind_size
    total_hybrid_plant_capacity_mw = solar_size + wind_size
    nameplate_mw = total_hybrid_plant_capacity_mw
    interconnection_size_mw = interconnection_size
    hopp_outputs, resource_filename_wind, resource_filename_solar \
        = run_hopp_calc(Site, scenario_description, bos_details, total_hybrid_plant_capacity_mw, solar_size_mw, wind_size_mw,
                    nameplate_mw, interconnection_size_mw, load_resource_from_file,
                    ppa_price, results_dir)


    print('Finished site number {}'.format(site_num))

    results_filename = all_run_dir+'_LCOH\{}_{}.txt'.format(site_lat,site_lon)
    np.savetxt(results_filename,[hopp_outputs['LCOH [$/kg]']])
    results_filename = all_run_dir+'_MTPD\{}_{}.txt'.format(site_lat,site_lon)
    np.savetxt(results_filename,[hopp_outputs['H2 Production [tonne/day]']])
    results_filename = all_run_dir+'_MWin\{}_{}.txt'.format(site_lat,site_lon)
    np.savetxt(results_filename,[hopp_outputs['Total Generation [MWe]']])
    results_filename = all_run_dir+'_MWout\{}_{}.txt'.format(site_lat,site_lon)
    np.savetxt(results_filename,[hopp_outputs['Sold to Grid [MWe]']])


def run_all_hybrid_calcs(site_details, scenario_descriptions, results_dir, load_resource_from_file, wind_size,
                         solar_size, hybrid_size, electrolyzer_size_mw, bos_details, ppa_price, solar_tracking_mode, hub_height,
                         correct_wind_speed_for_height, all_run_dir):
    """
    Performs a multi-threaded run of run_hybrid_calc for the given input parameters.
    Returns a dataframe result for all sites
    :param site_details: DataFrame containing site details for all sites to be analyzed,
    including site_nums, lat, long, wind resource filename and solar resource filename.
    :param scenario_description: Project scenario - e.g. 'Wind Only', 'Solar Only', 'Hybrid - Wind & Solar'.
    :param results_dir: path to results directory
    :param load_resource_from_file: (boolean) flag which determines whether
    :param wind_size: capacity in MW of wind plant.
    :param solar_size: capacity in MW of solar plant.
    :param hybrid_size: capacity in MW of hybrid plant.
    :param electrolyzer_size_mw: size in MW of electrolyzer.
    :param bos_details: contains bos details including type of analysis to conduct (cost/mw, json lookup, HybridBOSSE).
    :param ppa_price: ppa price in $(USD)
    :param solar_tracking_mode: solar tracking mode
    :param hub_height: hub height in meters.
    :param correct_wind_speed_for_height: (boolean) flag determining whether wind speed is extrapolated to hub height.
    :param all_run_dir: path to directory for results from all runs
    :return: DataFrame of results for run_hybrid_calc at all sites (save_all_runs)
    """
    # Establish output DataFrame
    #save_all_runs = pd.DataFrame()
    lats = []
    lons = []
    LCOEs = []

    # Combine all arguments to pass to run_hybrid_calc
    all_args = zip(site_details['year'], site_details['site_nums'], repeat(scenario_descriptions), repeat(results_dir),
                   repeat(load_resource_from_file),
                   site_details['wind_filenames'], site_details['solar_filenames'],
                   site_details['lat'], site_details['lon'],
                   repeat(wind_size), repeat(solar_size), repeat(hybrid_size), repeat(electrolyzer_size_mw),
                   repeat(bos_details), repeat(ppa_price),
                   repeat(solar_tracking_mode), repeat(hub_height),
                   repeat(correct_wind_speed_for_height), repeat(all_run_dir))

    # Run a multi-threaded analysis
    with multiprocessing.Pool(10) as p:
        p.starmap(run_hybrid_calc, all_args)

    # # Run a single-threaded analysis
    # for all_arg in all_args:
    #     run_hybrid_calc(*all_arg)

if __name__ == '__main__':

    # Set paths
    parent_path = os.path.abspath(os.path.dirname(__file__))
    print("Parent path: ", parent_path)
    results_dir = os.path.join(parent_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Establish Project Scenarios and Parameter Ranges:
    bos_details = dict()
    bos_details['BOSSource'] = 'BOSLookup'  # Cost/MW, BOSLookup, HybridBOSSE, HybridBOSSE_manual
    bos_details['BOSFile'] = 'UPDATED_BOS_Summary_Results.json'
    bos_details['BOSScenario'] = 'TBD in analysis'  # Will be set to Wind Only, Solar Only,
    # Variable Ratio Wind and Solar Greenfield, or Solar Addition
    bos_details['BOSScenarioDescription'] = ''  # Blank or 'Overbuild'
    bos_details['Modify Costs'] = True
    bos_details['wind_capex_reduction'] = 0
    bos_details['solar_capex_reduction'] = 0
    bos_details['wind_bos_reduction'] = 0
    bos_details['solar_bos_reduction'] = 0
    bos_details['wind_capex_reduction_hybrid'] = 0
    bos_details['solar_capex_reduction_hybrid'] = 0
    bos_details['wind_bos_reduction_hybrid'] = 0
    bos_details['solar_bos_reduction_hybrid'] = 0

    load_resource_from_file = True
    solar_from_file = True
    wind_from_file = True
    on_land_only = False
    in_usa_only = False  # Only use one of (in_usa / on_land) flags

    # Set Analysis Location and Details
    year = 2013
    lat_int = 30 # lattitute interval in degrees
    lon_int = 20 # longitude interval in degrees
    NE_vertex = np.array([43.3,-77.3]) # NE of Buffalo
    SE_vertex = np.array([40.3,-79.7]) # SE of Pittsburgh
    SW_vertex = np.array([24.6,-98.3]) # SW of Brownsville
    NW_vertex = np.array([48.0,-101.9]) # NW of Bismarck
    num_lon = round((NE_vertex[1]+SE_vertex[1]-NW_vertex[1]-SW_vertex[1])/2/lon_int)+1
    coords = np.array([[],[]]).transpose()
    for i in range(num_lon):
        N_vertex = NE_vertex+(NW_vertex-NE_vertex)*i/(num_lon-1)
        S_vertex = SE_vertex+(SW_vertex-SE_vertex)*i/(num_lon-1)
        num_lat = round((N_vertex[0]-S_vertex[0])/lat_int)+1
        for j in range(num_lat):
            coords = np.vstack([coords,N_vertex+(S_vertex-N_vertex)*j/(num_lat-1)])
    num_loc = np.shape(coords)[0]
    desired_lats = list(coords[:,0])
    desired_lons = list(coords[:,1])

    # Load locations from list - cancels out previous block
    num_files = 38
    lon_lat_name_list = ['ngcc'+str(i)+'.csv' for i in np.arange(1,num_files)]
    for lon_lat_name in lon_lat_name_list:
        lon_lats = pd.read_csv(lon_lat_name,header=None)
        desired_lats = list(lon_lats.values[:,1])
        desired_lons = list(lon_lats.values[:,0])

        # Load wind and solar resource files for location nearest desired lats and lons
        # NB this resource information will be overriden by API retrieved data if load_resource_from_file is set to False
        sitelist_name = 'filtered_site_details_{}_locs_{}_year'.format(num_loc, year)
        # sitelist_name = 'site_details.csv'
        if load_resource_from_file:
            # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
            site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, year, not_rect=True,\
                                                max_dist=.1)  # Return contains
            # site_details = filter_sites(site_details, location='usa only')
            site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))
        else:
            # Creates the site_details file containing grid of lats, lons, years, and wind and solar filenames (blank
            # - to force API resource download)
            if os.path.exists(sitelist_name):
                site_details = pd.read_csv(sitelist_name)
            else:
                site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, year, not_rect=True)
                # Filter to locations in USA - MOVE TO PARALLEL
                site_details = filter_sites(site_details, location='usa only')
                site_details.to_csv(sitelist_name)

        solar_tracking_mode = 'Fixed'  # Currently not making a difference
        ppa_prices = [0.00]  # [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        hub_height_options = [110]  # [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        correct_wind_speed_for_height = True
        electrolyzer_sizes = [1230]
        wind_sizes = [1530,1570,1490,1650,1410]
        solar_sizes = [740,700,780,620,860]
        hybrid_sizes = [wind_sizes[i] + solar_sizes[i] for i in range(len(wind_sizes))]

        for ppa_price in ppa_prices:
            for hub_height in hub_height_options:
                for electrolyzer_size_mw in electrolyzer_sizes:
                    for i, wind_size in enumerate(wind_sizes):
                        solar_size = solar_sizes[i]
                        hybrid_size = hybrid_sizes[i]

                        # Save results from all locations to folder
                        all_run_folder = 'All_Runs_WindSize_{}_MW_SolarSize_{}_MW'\
                            .format(wind_size, solar_size)
                        all_run_dir = os.path.join(parent_path, all_run_folder)
                        if not os.path.exists(all_run_dir+'_LCOH'):
                            os.mkdir(all_run_dir+'_LCOH')
                        if not os.path.exists(all_run_dir+'_MTPD'):
                            os.mkdir(all_run_dir+'_MTPD')
                        if not os.path.exists(all_run_dir+'_MWin'):
                            os.mkdir(all_run_dir+'_MWin')
                        if not os.path.exists(all_run_dir+'_MWout'):
                            os.mkdir(all_run_dir+'_MWout')

                        # Run hybrid calculation for all sites
                        # save_all_runs = 
                        run_all_hybrid_calcs(site_details, "greenfield", results_dir,
                                                            load_resource_from_file, wind_size,
                                                            solar_size, hybrid_size, electrolyzer_size_mw, bos_details,
                                                            ppa_price, solar_tracking_mode, hub_height,
                                                            correct_wind_speed_for_height, all_run_dir)
                    
                        # print(save_all_runs)
