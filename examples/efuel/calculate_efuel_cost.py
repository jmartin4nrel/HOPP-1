from hopp.simulation import HoppInterface
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from hopp.simulation.technologies.sites import SiteInfo, methanol_site
from examples.efuel.set_atb_year import set_atb_year
from examples.efuel.H2AModel_costs import H2AModel_costs
from examples.efuel.extract_cambium_data import set_cambium_inputs
from hopp.utilities import load_yaml
from pathlib import Path
import time

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string

H2O_price_tgal = 2.56 # 2020 $/Tgal
H2O_price_kg = H2O_price_tgal/1000*3.78

def calculate_efuel_cost(main_path: Path,
                         turndown_path: Path,
                         fuel: str='methanol',
                         reactor: str='CO2 hydrogenation',
                         catalyst: str=None,
                         pct_wind: float=100.,
                         pct_overbuild: float=0.,
                         dollar_year: int=2020,
                         startup_year: int=2020,
                         lat: float = 40.,
                         lon: float = -100.,
                         state: str = 'TX',
                         printout=False,
                         turndown=False,
                         grid_pricing=False,
                         wind_cap_pct=None,
                         pv_cap_pct=None):

    # Create a HOPP interface with the cost and lca information for all components loaded from a .yaml 
    hi = HoppInterface(main_path)

    # Set fuel, reactor, and catalyst
    getattr(hi.system,'fuel').value('fuel_produced',fuel)
    getattr(hi.system,'fuel').value('reactor_tech',reactor)
    if reactor == 'CO2 hydrogenation':
        catalyst = 'None'
    getattr(hi.system,'fuel').value('catalyst',catalyst)
    input_path = Path('inputs')
    output_path = Path('outputs')
    fuel_inputs = pd.read_csv(input_path/'Reactor_inputs_doe.csv',index_col=[0,1])
    cost_list = ['toc','toc_kg_s','foc_yr','foc_kg_s_yr','voc_kg','co2']
    for cost in cost_list:
        if cost in fuel_inputs.columns:
            setattr(hi.system.fuel.config.simple_fin_config,cost,fuel_inputs.loc[(reactor,catalyst),cost])
            setattr(hi.system.fuel._financial_model,cost,fuel_inputs.loc[(reactor,catalyst),cost])
    em_list = ['co2_kg_kg','h2o_kg_kg']
    for em in em_list:
        if em in fuel_inputs.columns:
            hi.system.fuel.config.lca[em] = fuel_inputs.loc[(reactor,catalyst),em]
    param_list = ['h2ratio','co2ratio']
    for param in param_list:
        if param in fuel_inputs.columns:
            hi.system.fuel.config.reaction_params[param] = fuel_inputs.loc[(reactor,catalyst),param]
    elec_ratio = fuel_inputs.loc[(reactor,catalyst),'kwh_kg']
    if 'RCC' in reactor:
        setattr(hi.system.co2.config,'capture_model','None')
        setattr(hi.system.tech_config.co2,'capture_model','None')

    # Correct year with ATB
    atb_scenario = "Moderate" # 'Advanced' # 
    hi = set_atb_year(hi, atb_scenario, startup_year, lat, lon)
    
    # Calculate co2 and h2 needed
    hi.system.fuel.simulate_flow(1)
    co2_kg_s = copy.deepcopy(np.mean(hi.system.fuel._system_model.input_streams_kg_s['carbon dioxide']))
    h2_kg_s = copy.deepcopy(np.mean(hi.system.fuel._system_model.input_streams_kg_s['hydrogen']))
    
    # Correct run H2A model to get electrolyzer efficiency and costs
    electrolyzer_cap_factor = 0.97
    h2_basis_year, h2_toc, h2_foc_yr, h2_WC_kg_h2o_kg_h2, h2_kWh_kg = H2AModel_costs(electrolyzer_cap_factor,
                                                                                       h2_kg_s*60*60*24,
                                                                                       startup_year)
    h2_elec_kw = h2_kg_s*h2_kWh_kg*3600
    meoh_kg_yr = hi.system.fuel.annual_mass_kg
    meoh_kw = meoh_kg_yr/8760*elec_ratio
    
    getattr(hi.system,'co2').value('co2_kg_s',co2_kg_s)
    # ngcc_cap =  hi.system.co2._system_model.ngcc_cap
    getattr(hi.system,'co2').value('system_capacity_kg_s',co2_kg_s)
    if hi.system.tech_config.co2.capture_model == 'None':
        hi.system.co2._financial_model.toc_kg_s = 0.
        hi.system.co2._financial_model.foc_kg_s_yr = 0.
        hi.system.co2._financial_model.voc_kg = 0.
        hi.system.tech_config.co2.lca['co2_kg_kg'] = 0.
        hi.system.tech_config.co2.lca['h2o_kg_kg'] = 0.
        hi.system.ng._system_model.annual_mass_kg = 0.
    hi.system.co2.simulate_flow(1)
    ng_kg_s = copy.deepcopy(np.mean(hi.system.co2._system_model.input_streams_kg_s['natural gas']))
    if "SMR" in reactor:
        ng_kg_s = 81.69883185*115104000/3365095697
    hi.system.ng._system_model.ng_kg_s = ng_kg_s
    getattr(hi.system,'ng').value('ng_kg_s',ng_kg_s)
    hi.system.tech_config.ng.ng_kg_s = ng_kg_s
    hi.system.ng.config.ng_kg_s = ng_kg_s
    hi.system.ng.ng_kg_s = ng_kg_s
    
    
    # Estimate number of turbines needed
    percent_wind = pct_wind
    percent_overbuild = pct_overbuild
    overbuild_elec_kw = (h2_elec_kw+meoh_kw)*(100+percent_overbuild)/100
    if wind_cap_pct:
        wind_cap_factor = wind_cap_pct/100
    else:
        wind_cap_factor = 0.45
    wind_cap_kw = overbuild_elec_kw*percent_wind/100/wind_cap_factor
    turb_rating_kw = getattr(hi.system,'wind').value('turb_rating')
    num_turbines = np.min([300,np.max([1,np.floor(wind_cap_kw/turb_rating_kw)])])

    # Widen site to match number of turbines needed
    # For site area: square with sides = sqrt of number of turbines times rotor diameter times ten
    d = hi.system.wind.rotor_diameter
    n = num_turbines
    side = 10*d*n**.5
    methanol_site['site_boundaries']['bounding_box'] = np.array([[0,0],[side,side]])
    methanol_site['site_boundaries']['verts'] = np.array([[0,0],[0,side],[side,side],[side,0]])
    methanol_site['site_boundaries']['verts_simple'] = np.array([[0,0],[0,side],[side,side],[side,0]])
    methanol_site['lat'] = lat
    methanol_site['lon'] = lon
    site = SiteInfo(
            methanol_site,
            solar_resource_file='',
            wind_resource_file='',
            grid_resource_file='',
            solar=True,
            wind=True,
            wave=False,
            hub_height=90.2,
        )
    hopp_config = load_yaml(hi.configuration)

    # set SiteInfo instance
    hopp_config["site"] = site
    if turndown:
        batt_kw = hi.system.battery.system_capacity_kw
        batt_kwh = hi.system.battery.system_capacity_kwh
    hopp_config["technologies"].pop("battery")

    # Create new instance of hopp interface with correct number of turbines
    hi = HoppInterface(hopp_config)

    # Correct year with ATB
    atb_scenario = "Moderate" # 'Advanced' # 
    hi = set_atb_year(hi, atb_scenario, startup_year, lat, lon)

    # Re-calculate wind power and finalize number of turbines
    getattr(hi.system,'wind').value('num_turbines',num_turbines)
    if wind_cap_pct:
        hi.system.wind.loaded_capacity_factor = wind_cap_factor
    hi.system.wind._financial_model.system_capacity_kw = hi.system.wind._system_model.Farm.system_capacity
    if not wind_cap_pct:
        hi.system.wind.simulate_power(1)
        wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
    wind_cap_kw = overbuild_elec_kw*percent_wind/100/wind_cap_factor
    num_turbines = np.min([300,np.max([1,np.floor(wind_cap_kw/turb_rating_kw)])])
    getattr(hi.system,'wind').value('num_turbines',num_turbines)
    if not wind_cap_pct:
        hi.system.wind.simulate_power(1)
        wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
    wind_cap_kw = num_turbines*turb_rating_kw
    hi.system.wind._financial_model.system_capacity_kw = wind_cap_kw
    percent_wind = wind_cap_kw*wind_cap_factor/overbuild_elec_kw*100
    if not wind_cap_pct:
        while percent_wind >= 100 and num_turbines>1:
            num_turbines = num_turbines-1
            getattr(hi.system,'wind').value('num_turbines',num_turbines)
            hi.system.wind.simulate_power(1)
            wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
            wind_cap_kw = num_turbines*turb_rating_kw
            hi.system.wind._financial_model.system_capacity_kw = wind_cap_kw
            percent_wind = wind_cap_kw*wind_cap_factor/overbuild_elec_kw*100


    # Set everything back to where it was
    cost_list = ['toc','toc_kg_s','foc_yr','foc_kg_s_yr','voc_kg']
    for cost in cost_list:
        if cost in fuel_inputs.columns:
            setattr(hi.system.fuel.config.simple_fin_config,cost,fuel_inputs.loc[(reactor,catalyst),cost])
            setattr(hi.system.fuel._financial_model,cost,fuel_inputs.loc[(reactor,catalyst),cost])
    em_list = ['co2_kg_kg','h2o_kg_kg']
    for em in em_list:
        if em in fuel_inputs.columns:
            hi.system.fuel.config.lca[em] = fuel_inputs.loc[(reactor,catalyst),em]
    param_list = ['h2ratio','co2ratio']
    for param in param_list:
        if param in fuel_inputs.columns:
            hi.system.fuel.config.reaction_params[param] = fuel_inputs.loc[(reactor,catalyst),param]
    elec_ratio = fuel_inputs.loc[(reactor,catalyst),'kwh_kg']
    if 'RCC' in reactor:
        setattr(hi.system.co2.config,'capture_model','None')
        setattr(hi.system.tech_config.co2,'capture_model','None')
    # getattr(hi.system,'co2').value('co2_kg_s',co2_kg_s)
    getattr(hi.system,'co2').value('system_capacity_kg_s',co2_kg_s)
    if hi.system.tech_config.co2.capture_model == 'None':
        hi.system.co2._financial_model.toc_kg_s = 0.
        hi.system.co2._financial_model.foc_kg_s_yr = 0.
        hi.system.co2._financial_model.voc_kg = 0.
        hi.system.tech_config.co2.lca['co2_kg_kg'] = 0.
        hi.system.tech_config.co2.lca['h2o_kg_kg'] = 0.
        hi.system.ng._system_model.annual_mass_kg = 0.
    hi.system.ng._system_model.ng_kg_s = ng_kg_s
    getattr(hi.system,'ng').value('ng_kg_s',ng_kg_s)
    hi.system.tech_config.ng.ng_kg_s = ng_kg_s
    hi.system.ng.config.ng_kg_s = ng_kg_s
    hi.system.ng.ng_kg_s = ng_kg_s

    # Calculate the (continuous) pv plant size needed based on an estimated capacity factor and the wind plant size
    percent_pv = 100-percent_wind
    if not pv_cap_pct:
        pv_cap_factor = 0.22
    else:
        pv_cap_factor = pv_cap_pct/100
        hi.system.pv.loaded_capacity_factor = pv_cap_factor
    pv_cap_kw = np.max([0.1,overbuild_elec_kw*percent_pv/100/pv_cap_factor])
    getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)
    if not pv_cap_pct:
        hi.system.pv.simulate_power(1)
        pv_cap_factor = hi.system.pv._system_model.Outputs.capacity_factor/100
        pv_cap_kw = np.max([0.1,overbuild_elec_kw*percent_pv/100/pv_cap_factor])
        getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)
        hi.system.pv.simulate_power(1)
        pv_cap_factor = hi.system.pv._system_model.Outputs.capacity_factor/100

    # Calculate the electrolyzer and interconnect size needed based on an estimated capacity factor
    electrolyzer_cap_factor = 0.97
    electrolyzer_cap_kw = h2_elec_kw/electrolyzer_cap_factor
    sales_cap_kw = wind_cap_kw+pv_cap_kw-electrolyzer_cap_kw-meoh_kw
    sales_cap_kw = max(1.,sales_cap_kw)
    if "SMR" in reactor:
        sales_cap_kw = elec_ratio*meoh_kg_yr/8760
    getattr(hi.system,'grid').value('interconnect_kw',sales_cap_kw+electrolyzer_cap_kw+meoh_kw)
    getattr(hi.system,'grid_sales').value('interconnect_kw',sales_cap_kw)
    getattr(hi.system,'grid_purchase').value('interconnect_kw',electrolyzer_cap_kw+meoh_kw)
    getattr(hi.system,'electrolyzer').value('system_capacity_kw',electrolyzer_cap_kw)
    setattr(hi.system.electrolyzer.config,'kwh_kg_h2',h2_kWh_kg)
    setattr(hi.system.electrolyzer._system_model,'kwh_kg_h2',h2_kWh_kg)
    setattr(hi.system.electrolyzer._financial_model,'input_dollar_yr',h2_basis_year)
    setattr(hi.system.electrolyzer._financial_model,'toc',h2_toc)
    setattr(hi.system.electrolyzer._financial_model,'toc_kw',None)
    setattr(hi.system.electrolyzer._financial_model,'foc_yr',h2_foc_yr)
    setattr(hi.system.electrolyzer._financial_model,'foc_kw_yr',None)
    water_cost_kwh = H2O_price_kg*h2_WC_kg_h2o_kg_h2/h2_kWh_kg
    setattr(hi.system.electrolyzer._financial_model,'voc_kwh',water_cost_kwh)
    if 'battery' in hi.system.technologies.keys():
        battery_pct_elec = 100
        battery_size_kw = battery_pct_elec/100*electrolyzer_cap_kw*electrolyzer_cap_factor
        battery_hrs = 8
        getattr(hi.system,'battery').value('system_capacity_kw',battery_size_kw)
        getattr(hi.system,'battery').value('system_capacity_kwh',battery_size_kw*battery_hrs)

    plant_life = 1 #years
    hi.simulate(plant_life)

    # Determine the capacity threshold at which grid electricity must be bought to meet electrolyzer demand
    cap_thresh = copy.deepcopy(electrolyzer_cap_factor)
    needed_kwh = (h2_elec_kw+meoh_kw)*8760
    total_kwh = (electrolyzer_cap_kw+meoh_kw)*8760
    if not wind_cap_pct and not pv_cap_pct:
        gen_profile = copy.deepcopy(hi.system.generation_profile['hybrid'])
        load_schedule = list(gen_profile)
    else:
        gen_profile = np.squeeze(pd.read_csv(input_path/'default_gen_profile.csv',header=None).values)
        default_kwh = np.sum(gen_profile)#1337238657.67414
        scale_kwh = (pv_cap_factor*pv_cap_kw + wind_cap_factor*wind_cap_kw)*8760
        gen_profile *= scale_kwh/default_kwh
        load_schedule = list(gen_profile)
    timestep_h = 8760/len(load_schedule)
    while total_kwh > needed_kwh:
        when_above = [i>=(electrolyzer_cap_kw+meoh_kw) for i in gen_profile]
        when_below = [i<=(electrolyzer_cap_kw*cap_thresh+meoh_kw) for i in gen_profile]
        when_in_between = list(np.logical_and(np.logical_not(when_above),np.logical_not(when_below)))
        total_kwh = sum(when_above)*(electrolyzer_cap_kw+meoh_kw)*timestep_h + \
                    sum(when_below)*(electrolyzer_cap_kw*cap_thresh+meoh_kw)*timestep_h + \
                    sum(np.multiply(gen_profile,when_in_between))*timestep_h
        if total_kwh > needed_kwh:
            cap_thresh -= 0.0001
        else:
            makeup_kwh = needed_kwh - sum(when_above)*(electrolyzer_cap_kw+meoh_kw)*timestep_h - \
                                        sum(np.multiply(gen_profile,when_in_between))*timestep_h
            makeup_kw = makeup_kwh/sum(when_below)/timestep_h
            cap_thresh = (makeup_kw-meoh_kw)/electrolyzer_cap_kw

    # Import cambium prices and emissions
    cambium_scenario = 'MidCase'
    hi = set_cambium_inputs(hi, cambium_scenario, startup_year, state)

    # Make electrolyzer/sales/purchase profiles
    sell_kw = [0.0]*8760
    buy_kw = [0.0]*8760
    for i, gen in enumerate(load_schedule):
        if when_above[i]:
            sell_kw[i] = electrolyzer_cap_kw+meoh_kw-gen
            load_schedule[i] = electrolyzer_cap_kw+meoh_kw
        elif when_below[i]:
            load_schedule[i] = electrolyzer_cap_kw*cap_thresh+meoh_kw
            buy_kw[i] = electrolyzer_cap_kw*cap_thresh+meoh_kw-gen
    hi.system.electrolyzer.generation_profile = [i-meoh_kw for i  in load_schedule]
    hi.system.grid_sales.generation_profile = sell_kw
    hi.system.grid_purchase.generation_profile = buy_kw

    # Simulate plant for 30 years, getting curtailment (will be sold to grid) and missed load (will be purchased from grid)
    plant_life = 1 #years
    hi.simulate(plant_life)

    

    if turndown:
        site = SiteInfo(
                methanol_site,
                solar_resource_file=hi.system.site.solar_resource_file,
                wind_resource_file=hi.system.site.wind_resource_file,
                grid_resource_file=hi.system.site.grid_resource_file,
                desired_schedule=[i/1000 for i in load_schedule],
                solar=True,
                wind=True,
                wave=False
            )

        hopp_config = load_yaml(turndown_path)
        # set SiteInfo instance
        hopp_config["site"] = site
        hopp_config["technologies"]["wind"]["num_turbines"] = num_turbines
        hopp_config["technologies"]["pv"]["system_capacity_kw"] = pv_cap_kw
        hopp_config["technologies"]["battery"]["system_capacity_kw"] = batt_kw
        hopp_config["technologies"]["battery"]["system_capacity_kwh"] = batt_kwh

        hi_batt = HoppInterface(hopp_config)

        # Correct year with ATB
        atb_scenario = "Moderate" # 'Advanced' # 
        hi_batt = set_atb_year(hi_batt, atb_scenario, startup_year, lat, lon)

    
        if not grid_pricing:
            hi_batt.system.dispatch_factors = (1.0,)*8760

        hi_batt.simulate(project_life=1)

    if not grid_pricing:
        hi.system.dispatch_factors = (1.0,)*8760
    
    hybrid_plant = hi.system
    # solar_plant_power = np.array(hybrid_plant.pv.generation_profile)
    # wind_plant_power = np.array(hybrid_plant.wind.generation_profile)
    renewable_generation_profile = gen_profile#solar_plant_power + wind_plant_power
    electrolyzer_profile = np.array(hybrid_plant.electrolyzer.generation_profile)+meoh_kw


    if turndown:
        batt_plant = hi_batt.system
        if 'battery' in hi.system.technologies.keys():
            extra_cap_kw = hi.system.battery.system_capacity_kw
            electrolyzer_cap_kw += extra_cap_kw
            getattr(hi.system,'electrolyzer').value('system_capacity_kw',electrolyzer_cap_kw)
            batt_power = np.array(batt_plant.battery.generation_profile)
            electrolyzer_extra_profile = electrolyzer_profile-batt_power
            hi.system.electrolyzer.generation_profile = list(electrolyzer_extra_profile)
            batt_bought = np.maximum(0,electrolyzer_extra_profile-renewable_generation_profile)
            batt_sold = -np.maximum(0,renewable_generation_profile-electrolyzer_extra_profile)
            getattr(hi.system,'battery').value('system_capacity_kw',0.00001)
        else:
            batt_bought = np.maximum(0,electrolyzer_profile-renewable_generation_profile)
            batt_sold = -np.maximum(0,renewable_generation_profile-electrolyzer_profile)
    else:
        batt_bought = np.maximum(0,electrolyzer_profile-renewable_generation_profile)
        batt_sold = -np.maximum(0,renewable_generation_profile-electrolyzer_profile)
        
    hi.system.grid_purchase.generation_profile = list(batt_bought)
    hi.system.grid_sales.generation_profile = list(batt_sold)

    if "SMR" in reactor:
        smr_sold = -np.ones(8760)*elec_ratio*meoh_kg_yr/8760
        hi.system.grid_sales.generation_profile = list(smr_sold)
    
    hi.simulate(1)

    if printout:
        out_list = []
        print("Percent wind: {:.1f}%   Percent overbuild: {:.1f}%".format(percent_wind,pct_overbuild))
        out_fn = 'lat_{:.3f}_lon_{:.3f}_reactor_{}_catalyst_{}_wind_{:.1f}_over_{:.1f}.csv'.format(lat,lon,reactor,catalyst.replace('/','-'),percent_wind,percent_overbuild)
        print("Levelized cost of methanol (LCOM), $/kg: {:.3f}".format(hi.system.lc))
        for tech in hi.system.lc_breakdown.keys():
            print(tech+': {:.3f}'.format(hi.system.lc_breakdown[tech]))
            out_list.append(hi.system.lc_breakdown[tech])
        print("Fraction of electricity to methanol: {:.5f}".format(meoh_kw/(meoh_kw+h2_elec_kw)))
        out_list.append(meoh_kw/(meoh_kw+h2_elec_kw))
        print("Carbon Intensity (CI), kg/kg-MeOH: {:.3f}".format(hi.system.lca['co2_kg_kg']))
        for tech in hi.system.lca_breakdown.keys():
            print(tech+': {:.3f}'.format(hi.system.lca_breakdown[tech]['co2_kg_kg']))
            out_list.append(hi.system.lca_breakdown[tech]['co2_kg_kg'])
        print("Water Consumption (WC), kg/kg-MeOH: {:.3f}".format(hi.system.lca['h2o_kg_kg']))
        for tech in hi.system.lca_breakdown.keys():
            print(tech+': {:.3f}'.format(hi.system.lca_breakdown[tech]['h2o_kg_kg']))
            out_list.append(hi.system.lca_breakdown[tech]['h2o_kg_kg'])
        np.savetxt(output_path/out_fn,np.array(out_list))

    return hi.system.lc, hi.system.lca['co2_kg_kg'], hi.system.lca['h2o_kg_kg'], wind_cap_factor, pv_cap_factor