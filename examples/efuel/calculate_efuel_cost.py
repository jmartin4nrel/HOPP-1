from hopp.simulation import HoppInterface
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from hopp.simulation.technologies.sites import SiteInfo, methanol_site
from hopp.utilities import load_yaml

def calculate_efuel_cost(pct_wind, pct_overbuild, lat, lon):

    # Create a HOPP interface with the cost and lca information for all components loaded from a .yaml 
    hi = HoppInterface("./08-wind-solar-electrolyzer-fuel.yaml")

    # Set system losses

    hi.system.pv._system_model.SystemDesign.dc_ac_ratio = 1.28
    hi.system.pv._system_model.SystemDesign.losses = 14.3

    hi.system.wind._system_model.Losses.avail_bop_loss = 0
    hi.system.wind._system_model.Losses.avail_grid_loss = 0
    hi.system.wind._system_model.Losses.avail_turb_loss = 0
    hi.system.wind._system_model.Losses.elec_eff_loss = 0
    hi.system.wind._system_model.Losses.elec_parasitic_loss = 0
    hi.system.wind._system_model.Losses.env_degrad_loss =0
    hi.system.wind._system_model.Losses.env_env_loss = 0
    hi.system.wind._system_model.Losses.env_icing_loss = 0
    hi.system.wind._system_model.Losses.ops_env_loss = 0
    hi.system.wind._system_model.Losses.ops_grid_loss = 0
    hi.system.wind._system_model.Losses.ops_load_loss = 0
    hi.system.wind._system_model.Losses.turb_generic_loss = 0
    hi.system.wind._system_model.Losses.turb_hysteresis_loss = 0
    hi.system.wind._system_model.Losses.turb_perf_loss = 0
    hi.system.wind._system_model.Losses.turb_specific_loss = 0
    hi.system.wind._system_model.Losses.wake_ext_loss = 0
    
    # Calculate electricity needed
    hi.system.fuel.simulate_flow(1)
    total_elec_kw = copy.deepcopy(np.mean((hi.system.fuel._system_model.input_streams_kw['electricity'])))
    
    co2_kg_s = copy.deepcopy(np.mean(hi.system.fuel._system_model.input_streams_kg_s['carbon dioxide']))
    getattr(hi.system,'co2').value('co2_kg_s',co2_kg_s)
    if hi.system.tech_config.co2.capture_model == 'None':
        hi.system.co2._financial_model.voc_kg = 0.
        hi.system.tech_config.co2.lca['co2_kg_kg'] = 0.
        hi.system.ng._system_model.annual_mass_kg = 0.
    hi.system.co2.simulate_flow(1)
    ng_kg_s = copy.deepcopy(np.mean(hi.system.co2._system_model.input_streams_kg_s['natural gas']))
    hi.system.ng._system_model.ng_kg_s = ng_kg_s
    getattr(hi.system,'ng').value('ng_kg_s',ng_kg_s)
    hi.system.tech_config.ng.ng_kg_s = ng_kg_s
    hi.system.ng.config.ng_kg_s = ng_kg_s
    hi.system.ng.ng_kg_s = ng_kg_s
    
    
    # Estimate number of turbines needed
    percent_wind = pct_wind
    percent_overbuild = pct_overbuild
    overbuild_elec_kw = total_elec_kw*(100+percent_overbuild)/100
    wind_cap_factor = 0.45
    wind_cap_kw = overbuild_elec_kw*percent_wind/100/wind_cap_factor
    turb_rating_kw = getattr(hi.system,'wind').value('turb_rating')
    num_turbines = np.max([1,np.floor(wind_cap_kw/turb_rating_kw)])

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
        )
    hopp_config = load_yaml(hi.configuration)
    # set SiteInfo instance
    hopp_config["site"] = site

    # Create new instance of hopp interface with correct number of turbines
    hi = HoppInterface(hopp_config)

    hi.system.pv._system_model.SystemDesign.dc_ac_ratio = 1.28
    hi.system.pv._system_model.SystemDesign.losses = 14.3

    hi.system.wind._system_model.Losses.avail_bop_loss = 0
    hi.system.wind._system_model.Losses.avail_grid_loss = 0
    hi.system.wind._system_model.Losses.avail_turb_loss = 0
    hi.system.wind._system_model.Losses.elec_eff_loss = 0
    hi.system.wind._system_model.Losses.elec_parasitic_loss = 0
    hi.system.wind._system_model.Losses.env_degrad_loss =0
    hi.system.wind._system_model.Losses.env_env_loss = 0
    hi.system.wind._system_model.Losses.env_icing_loss = 0
    hi.system.wind._system_model.Losses.ops_env_loss = 0
    hi.system.wind._system_model.Losses.ops_grid_loss = 0
    hi.system.wind._system_model.Losses.ops_load_loss = 0
    hi.system.wind._system_model.Losses.turb_generic_loss = 0
    hi.system.wind._system_model.Losses.turb_hysteresis_loss = 0
    hi.system.wind._system_model.Losses.turb_perf_loss = 0
    hi.system.wind._system_model.Losses.turb_specific_loss = 0
    hi.system.wind._system_model.Losses.wake_ext_loss = 0

    # Re-calculate wind power and finalize number of turbines
    getattr(hi.system,'wind').value('num_turbines',num_turbines)
    hi.system.wind._financial_model.system_capacity_kw = hi.system.wind._system_model.Farm.system_capacity
    hi.system.wind.simulate_power(1)
    wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
    wind_cap_kw = overbuild_elec_kw*percent_wind/100/wind_cap_factor
    num_turbines = np.max([1,np.floor(wind_cap_kw/turb_rating_kw)])
    getattr(hi.system,'wind').value('num_turbines',num_turbines)
    hi.system.wind.simulate_power(1)
    wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
    wind_cap_kw = num_turbines*turb_rating_kw
    hi.system.wind._financial_model.system_capacity_kw = wind_cap_kw
    percent_wind = wind_cap_kw*wind_cap_factor/overbuild_elec_kw*100
    # while percent_wind >= 100:
    #     num_turbines = num_turbines-1
    #     getattr(hi.system,'wind').value('num_turbines',num_turbines)
    #     hi.system.wind.simulate_power(1)
    #     wind_cap_factor = getattr(hi.system,'wind').value('capacity_factor')/100
    #     wind_cap_kw = num_turbines*turb_rating_kw
    #     hi.system.wind._financial_model.system_capacity_kw = wind_cap_kw
    #     percent_wind = wind_cap_kw*wind_cap_factor/overbuild_elec_kw*100

    # Set everything back to where it was
    getattr(hi.system,'co2').value('co2_kg_s',co2_kg_s)
    if hi.system.tech_config.co2.capture_model == 'None':
        hi.system.co2._financial_model.voc_kg = 0.
        hi.system.tech_config.co2.lca['co2_kg_kg'] = 0.
        hi.system.ng._system_model.annual_mass_kg = 0.
    hi.system.ng._system_model.ng_kg_s = ng_kg_s
    getattr(hi.system,'ng').value('ng_kg_s',ng_kg_s)
    hi.system.tech_config.ng.ng_kg_s = ng_kg_s
    hi.system.ng.config.ng_kg_s = ng_kg_s
    hi.system.ng.ng_kg_s = ng_kg_s

    # Calculate the (continuous) pv plant size needed based on an estimated capacity factor and the wind plant size
    percent_pv = 100-percent_wind
    pv_cap_factor = 0.22
    pv_cap_kw = np.max([0.1,overbuild_elec_kw*percent_pv/100/pv_cap_factor])
    getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)
    hi.system.pv.simulate_power(1)
    pv_cap_factor = hi.system.pv._system_model.Outputs.capacity_factor/100
    pv_cap_kw = np.max([0.1,overbuild_elec_kw*percent_pv/100/pv_cap_factor])
    getattr(hi.system,'pv').value('system_capacity_kw',pv_cap_kw)
    hi.system.pv.simulate_power(1)
    pv_cap_factor = hi.system.pv._system_model.Outputs.capacity_factor/100

    # Calculate the electrolyzer and interconnect size needed based on an estimated capacity factor
    electrolyzer_cap_factor = 0.97
    electrolyzer_cap_kw = total_elec_kw/electrolyzer_cap_factor
    sales_cap_kw = wind_cap_kw+pv_cap_kw-electrolyzer_cap_kw
    sales_cap_kw = max(1.,sales_cap_kw)
    getattr(hi.system,'grid').value('interconnect_kw',np.max([sales_cap_kw,electrolyzer_cap_kw]))
    getattr(hi.system,'grid_sales').value('interconnect_kw',sales_cap_kw)
    getattr(hi.system,'grid_purchase').value('interconnect_kw',electrolyzer_cap_kw)
    getattr(hi.system,'electrolyzer').value('system_capacity_kw',electrolyzer_cap_kw)
    if 'battery' in hi.system.technologies.keys():
        battery_pct_elec = 100
        battery_size_kw = battery_pct_elec/100*electrolyzer_cap_kw*electrolyzer_cap_factor
        battery_hrs = 8
        getattr(hi.system,'battery').value('system_capacity_kw',battery_size_kw)
        getattr(hi.system,'battery').value('system_capacity_kwh',battery_size_kw*battery_hrs)

    plant_life = 1 #years
    hi.simulate(plant_life)

    # Determine the capacity threshold at which grid electricity must be bought to meet electrolyzer demand
    cap_thresh = electrolyzer_cap_factor
    needed_kwh = electrolyzer_cap_factor*electrolyzer_cap_kw*8760
    total_kwh = electrolyzer_cap_kw*8760
    timestep_h = 8760/len(hi.system.generation_profile['hybrid'])
    while total_kwh > needed_kwh:
        when_above = [i>=electrolyzer_cap_kw for i in hi.system.generation_profile['hybrid']]
        when_below = [i<=electrolyzer_cap_kw*cap_thresh for i in hi.system.generation_profile['hybrid']]
        when_in_between = list(np.logical_and(np.logical_not(when_above),np.logical_not(when_below)))
        total_kwh = sum(when_above)*electrolyzer_cap_kw*timestep_h + \
                    sum(when_below)*electrolyzer_cap_kw*cap_thresh*timestep_h + \
                    sum(np.multiply(hi.system.generation_profile['hybrid'],when_in_between))*timestep_h
        if total_kwh > needed_kwh:
            cap_thresh -= 0.001
        else:
            makeup_kwh = needed_kwh - sum(when_above)*electrolyzer_cap_kw*timestep_h - \
                                        sum(np.multiply(hi.system.generation_profile['hybrid'],when_in_between))*timestep_h
            makeup_kw = makeup_kwh/sum(when_below)/timestep_h
            cap_thresh = makeup_kw/electrolyzer_cap_kw

    # Make electrolyzer/sales/purchase profiles
    sell_kw = [0.0]*8760
    buy_kw = [0.0]*8760
    load_schedule = list(hi.system.generation_profile['hybrid'])
    for i, gen in enumerate(load_schedule):
        if when_above[i]:
            sell_kw[i] = electrolyzer_cap_kw-gen
            load_schedule[i] = electrolyzer_cap_kw
        elif when_below[i]:
            load_schedule[i] = electrolyzer_cap_kw*cap_thresh
            buy_kw[i] = electrolyzer_cap_kw*cap_thresh-gen
    hi.system.electrolyzer.generation_profile = load_schedule
    hi.system.grid_sales.generation_profile = sell_kw
    hi.system.grid_purchase.generation_profile = buy_kw

    # Simulate plant for 30 years, getting curtailment (will be sold to grid) and missed load (will be purchased from grid)
    plant_life = 1 #years
    hi.simulate(plant_life)

    # site = SiteInfo(
    #         methanol_site,
    #         solar_resource_file=hi.system.site.solar_resource_file,
    #         wind_resource_file=hi.system.site.wind_resource_file,
    #         grid_resource_file=hi.system.site.grid_resource_file,
    #         desired_schedule=[i/1000 for i in load_schedule],
    #         solar=True,
    #         wind=True,
    #         wave=False
    #     )

    # hopp_config = load_yaml("./09-methanol-battery.yaml")
    # # set SiteInfo instance
    # hopp_config["site"] = site
    # hopp_config["technologies"]["wind"]["num_turbines"] = num_turbines
    # hopp_config["technologies"]["pv"]["system_capacity_kw"] = pv_cap_kw
    # if 'battery' in hi.system.technologies.keys():
    #     hopp_config["technologies"]["battery"]["system_capacity_kw"] = hi.system.battery.system_capacity_kw
    #     hopp_config["technologies"]["battery"]["system_capacity_kwh"] = hi.system.battery.system_capacity_kwh
    # else:
    #     hopp_config["technologies"].pop("battery")

    # hi_batt = HoppInterface(hopp_config)

    hi.system.dispatch_factors = (1.0,)*8760

    # hi_batt.system.dispatch_factors = (1.0,)*8760

    # hi_batt.system.pv._system_model.SystemDesign.dc_ac_ratio = hi.system.pv._system_model.SystemDesign.dc_ac_ratio
    # hi_batt.system.pv._system_model.SystemDesign.losses = hi.system.pv._system_model.SystemDesign.losses

    # hi_batt.system.wind._system_model.Losses.avail_bop_loss = hi.system.wind._system_model.Losses.avail_bop_loss
    # hi_batt.system.wind._system_model.Losses.avail_grid_loss = hi.system.wind._system_model.Losses.avail_grid_loss
    # hi_batt.system.wind._system_model.Losses.avail_turb_loss = hi.system.wind._system_model.Losses.avail_turb_loss
    # hi_batt.system.wind._system_model.Losses.elec_eff_loss = hi.system.wind._system_model.Losses.elec_eff_loss
    # hi_batt.system.wind._system_model.Losses.elec_parasitic_loss = hi.system.wind._system_model.Losses.elec_parasitic_loss
    # hi_batt.system.wind._system_model.Losses.env_degrad_loss = hi.system.wind._system_model.Losses.env_degrad_loss
    # hi_batt.system.wind._system_model.Losses.env_env_loss = hi.system.wind._system_model.Losses.env_env_loss
    # hi_batt.system.wind._system_model.Losses.env_icing_loss = hi.system.wind._system_model.Losses.env_icing_loss
    # hi_batt.system.wind._system_model.Losses.ops_env_loss = hi.system.wind._system_model.Losses.ops_env_loss
    # hi_batt.system.wind._system_model.Losses.ops_grid_loss = hi.system.wind._system_model.Losses.ops_grid_loss
    # hi_batt.system.wind._system_model.Losses.ops_load_loss = hi.system.wind._system_model.Losses.ops_load_loss
    # hi_batt.system.wind._system_model.Losses.turb_generic_loss = hi.system.wind._system_model.Losses.turb_generic_loss
    # hi_batt.system.wind._system_model.Losses.turb_hysteresis_loss = hi.system.wind._system_model.Losses.turb_hysteresis_loss
    # hi_batt.system.wind._system_model.Losses.turb_perf_loss = hi.system.wind._system_model.Losses.turb_perf_loss
    # hi_batt.system.wind._system_model.Losses.turb_specific_loss = hi.system.wind._system_model.Losses.turb_specific_loss
    # hi_batt.system.wind._system_model.Losses.wake_ext_loss = hi.system.wind._system_model.Losses.wake_ext_loss

    hybrid_plant = hi.system
    solar_plant_power = np.array(hybrid_plant.pv.generation_profile)
    wind_plant_power = np.array(hybrid_plant.wind.generation_profile)
    renewable_generation_profile = solar_plant_power + wind_plant_power
    sold_power = np.array(hybrid_plant.grid_sales.generation_profile)
    bought_power = np.array(hybrid_plant.grid_purchase.generation_profile)
    electrolyzer_profile = np.array(hybrid_plant.electrolyzer.generation_profile)


    # batt_plant = hi_batt.system
    # batt_solar_plant_power = np.array(batt_plant.pv.generation_profile)
    # batt_wind_plant_power = np.array(batt_plant.wind.generation_profile)
    # if 'battery' in hi.system.technologies.keys():
    #     extra_cap_kw = hi.system.battery.system_capacity_kw
    #     electrolyzer_cap_kw += extra_cap_kw
    #     getattr(hi.system,'electrolyzer').value('system_capacity_kw',electrolyzer_cap_kw)
    #     batt_power = np.array(batt_plant.battery.generation_profile)
    #     batt_SOC = np.array(batt_plant.battery.outputs.SOC)
    #     electrolyzer_extra_profile = electrolyzer_profile-batt_power
    #     hi.system.electrolyzer.generation_profile = list(electrolyzer_extra_profile)
    #     batt_bought = np.maximum(0,electrolyzer_extra_profile-renewable_generation_profile)
    #     batt_sold = -np.maximum(0,renewable_generation_profile-electrolyzer_extra_profile)
    #     getattr(hi.system,'battery').value('system_capacity_kw',0.00001)
    # else:
    #     batt_power = np.array([0.0]*8760)
    #     batt_SOC = np.array([0.0]*8760)
    batt_bought = np.maximum(0,electrolyzer_profile-renewable_generation_profile)
    batt_sold = -np.maximum(0,renewable_generation_profile-electrolyzer_profile)
        
    # batt_plant_power = np.add(renewable_generation_profile,batt_power)

    hi.system.grid_purchase.generation_profile = list(batt_bought)
    hi.system.grid_sales.generation_profile = list(batt_sold)
    
    hi.simulate(1)

    print("Percent wind: {:.1f}%   Percent overbuild: {:.1f}%".format(percent_wind,pct_overbuild))
    print("Levelized cost of methanol (LCOM), $/kg: {:.3f}".format(hi.system.lc))
    for tech in hi.system.lc_breakdown.keys():
        print(tech+': {:.3f}'.format(hi.system.lc_breakdown[tech]))
    # print("Carbon Intensity (CI), kg/kg-MeOH: {:.3f}".format(hi.system.lca['co2_kg_kg']))
    # for tech in hi.system.lca_breakdown.keys():
    #     print(tech+': {:.3f}'.format(hi.system.lca_breakdown[tech]['co2_kg_kg']))

    return hi.system.lc