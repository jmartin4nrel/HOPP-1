import os
import re
from pathlib import Path
import importlib
import json
import numpy as np
import pandas as pd
import copy as copy
import matplotlib.pyplot as plt
import matplotlib.patches as patch

from hybrid.hybrid_simulation import HybridSimulation
from hybrid.sites import SiteInfo
from hybrid.resource import WindResource
from hybrid.keys import set_nrel_key_dot_env

# Set API key
set_nrel_key_dot_env()

def validate_asset(asset_path, config, period_file, years=[2020], plot_val=False):

    # Set sub-directories
    res_path = asset_path/'resource'
    sys_path = asset_path/'system'
    sts_path = asset_path/'status'
    gen_path = asset_path/'generation'
    
    # Import resource data
    asset = asset_path.parts[-1]
    module = 'hybrid.validation.asset_data_parsing.{}.parse_resource_data'.format(asset)
    parse_resource_data = importlib.import_module(module)
    res_fps = {}
    for tech in os.listdir(res_path):
        # Find resource data and parse into format needed
        res_subpath = res_path/tech
        parse_tech = getattr(parse_resource_data, 'parse_{}'.format(tech))
        if os.listdir(res_subpath)[0] == 'UNAVAILABLE':
            res_fps[tech] = {i:'' for i in years}
        else:
            res_fps[tech] = parse_tech(res_subpath, years)
            
    # Find # of "parts" needed to validate ("_pt_#_of_#" on end of config directory)
    hybrid_files = os.listdir(sys_path/'hybrid')
    hybrid_subconfigs = []
    hybrid_suffixes = []
    for file in hybrid_files:
        if config in file:
            hybrid_subconfigs.append(file)
            hyphen_locs = [i.end() for i in re.finditer('_',file)]
            if len(hyphen_locs) > 0:
                hybrid_suffixes.append(file[hyphen_locs[-1]:])
    # Check that # of "parts" found matches the "of_#"
    if len(hybrid_suffixes)>0:
        num_parts = int(hybrid_suffixes[0])
    if len(np.unique(hybrid_suffixes)) != 1 or len(hybrid_suffixes) != num_parts:
        raise FileNotFoundError('Did not find all parts of system configuration, check {}'.format(str(sys_path)))
    else:
        sys_info = {} # Dict of system info to be loaded from .jsons

    # Load info needed to make Sites and HybridSimulations and set model constants
    dicts_to_load = {'site':'site.json',
                     'tech':'technologies.json',
                     'cons':'system_constants.json'}
    for i in range(num_parts):
        subconfig = hybrid_subconfigs[i]
        sys_info[subconfig] = {}
        sys_subpath = sys_path/'hybrid'/subconfig
        for key, file in dicts_to_load.items():
            with open(sys_subpath/file, 'r') as fp:
                sys_info[subconfig][key] = json.load(fp)

    # Create SiteInfos and HybridSimulations
    # TODO: resource files, hub heights
    sites = {}
    hybrids = {}
    for subconfig, sub_dict in sys_info.items():
        site_dict = sub_dict['site']
        tech_dict = sub_dict['tech']
        sites[subconfig] = {}
        hybrids[subconfig] = {}
        for year in years:
            
            # Create SiteInfo
            if 'pv' in res_fps.keys():
                solar_fp = str(res_fps['pv'][year])
            else:
                solar_fp = ''
            if 'wind' in res_fps.keys():
                wind_fp = str(res_fps['wind'][year])
            else:
                wind_fp = ''
            if 'wind' in tech_dict.keys():
                hub_ht = tech_dict['wind']['hub_height']
                site = SiteInfo(site_dict, solar_fp, wind_fp, hub_height=hub_ht)
            else:
                site = SiteInfo(site_dict, solar_fp, wind_fp)
            sites[subconfig][year] = site
        
            # Create HybridSimulation
            technologies = copy.deepcopy(tech_dict)
            iconn_kw = technologies.pop('interconnect_kw')
            hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=iconn_kw)
            hybrids[subconfig][year] = hybrid_plant

            # Set model constants
            cons_dict = sub_dict['cons']
            constants = copy.deepcopy(cons_dict)
            if 'wind' in constants.keys():
                turb_model = constants['wind'].pop('turbine model')
                wind_power_curve = turb_model+'_Powercurve_10min.csv'
                curve_data = pd.read_csv(sys_path/'wind'/wind_power_curve)
                wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
                curve_power = curve_data['Power [kW]']
                hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
                hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power
            for tech, values in constants.items():
                for key, value in values.items():
                    getattr(hybrid_plant, tech).value(key, value)

    # Process status info #TODO: make this work not just for Flatirons example!
    module = 'hybrid.validation.asset_data_parsing.{}.parse_status_info'.format(asset)
    parse_status_info = importlib.import_module(module)
    parse_status = getattr(parse_status_info, 'parse_status')
    good_period_fp, status_fp, yaw_mismatch_fp = parse_status(sts_path, period_file, years)

    # Get yaw mismatch
    yaw_df = pd.read_json(yaw_mismatch_fp)
    dir_label = 'Wind Direction [deg]'
    mismatch_label = 'Yaw Mismatch [deg]'

    # Down-select periods where generation data matches resource data
    period_df = pd.read_csv(good_period_fp)
    pv_starts = pd.DatetimeIndex(period_df.loc[:,'PV Starts'])
    pv_stops = pd.DatetimeIndex(period_df.loc[:,'PV Stops'])
    good_pv_starts = [i in years for i in pv_starts.year]
    pv_starts = pv_starts[good_pv_starts]
    pv_stops = pv_stops[good_pv_starts]
    wind_starts = pd.DatetimeIndex(period_df.loc[:,'Wind Starts'])
    wind_stops = pd.DatetimeIndex(period_df.loc[:,'Wind Stops'])
    good_wind_starts = [i in years for i in wind_starts.year]
    wind_starts = wind_starts[good_wind_starts]
    wind_stops = wind_stops[good_wind_starts]

    # Read in processed status info (hours where status codes are valid)
    status_df = pd.read_json(Path(str(status_fp)))
    
    # Import generation data
    module = 'hybrid.validation.asset_data_parsing.{}.parse_generation_data'.format(asset)
    parse_generation_data = importlib.import_module(module)
    gen_fps = {}
    for tech in os.listdir(gen_path):
        # Find generation data and parse into format needed
        gen_subpath = gen_path/tech
        parse_tech = getattr(parse_generation_data, 'parse_{}'.format(tech))
        gen_fps[tech] = parse_tech(gen_subpath, years)
            
    use_dir = True
    use_status = True

    # Simulate year by year
    good_pv_gen = pd.DataFrame()
    good_pv_tun = pd.DataFrame()
    good_wind_gen = pd.DataFrame()
    good_wind_tun = pd.DataFrame()
    pv_tun_all = []
    wind_tun_all = []
    pv_gen_all = []
    wind_gen_all = []
    times_all = []
    poa_all = []
    wind_speed_all = []
    good_pv_inds_all = []
    good_wind_inds_all = []
    min_angle = 243
    max_angle = 310
    for subconfig in hybrid_subconfigs:
        for i, year in enumerate(years):
        
            # Pull hybrid plant out of dict
            hybrid = hybrids[subconfig][year]
            
            # Have to change pressure to sea level!
            hub_ht = hybrid.wind._system_model.Turbine.wind_turbine_hub_ht
            NewWindRes = WindResource(hybrid.site.lat,hybrid.site.lon,year,hub_ht,filepath=res_fps['wind'][year])
            for j in range(len(NewWindRes.data['data'])):
                NewWindRes.data['data'][j][1] = 1
            hybrid.wind._system_model.Resource.wind_resource_data = NewWindRes.data
            wind_speed_all.extend([i[3] for i in NewWindRes.data['data']])
        
            # Simulate power for 1 year
            hybrid.simulate_power(1)

            # Pull out plane-of-array irradiance
            poa_all.extend([i for i in getattr(hybrid,'pv').value('poa')])
            
            # Pull out simulated generation profiles
            pv_gen = hybrid.pv.generation_profile
            pv_gen_all.extend(pv_gen)
            wind_gen = hybrid.wind.generation_profile
            wind_gen_all.extend(wind_gen)

            # Load actual generation data
            pv_tun = pd.read_csv(gen_fps['pv'][year])
            wind_tun = pd.read_csv(gen_fps['wind'][year])
            pv_tun_P = pv_tun['P [kW]'].values
            pv_tun_Q = pv_tun['Q [kVA]'].values
            wind_tun_P = wind_tun['P [kW]'].values
            wind_tun_Q = wind_tun['Q [kVA]'].values
            pv_tun_P = [np.max([i,0]) for i in pv_tun_P]
            wind_tun_P = [np.max([i,0]) for i in wind_tun_P]
            pv_tun_S = np.sqrt(np.add(np.square(pv_tun_P),np.square(pv_tun_Q)))
            wind_tun_S = np.sqrt(np.add(np.square(wind_tun_P),np.square(wind_tun_Q)))
            pv_tun_S = [np.max([i,0]) for i in pv_tun_S]
            wind_tun_S = [np.max([i,0]) for i in wind_tun_S]
            pv_tun_all.extend(pv_tun_P)
            wind_tun_all.extend(wind_tun_P)

            if year % 4 == 0:
                # Take out leap day
                times = pd.date_range(start=str(year)+'-01-01 00:30:00',periods=8784,freq='H')
                times = times[:1416].union(times[1440:])
            else:
                times = pd.date_range(start=str(year)+'-01-01 00:30:00',periods=8760,freq='H')
            times_all.extend(times)
            
            # Get good periods
            pv_starts_year = pv_starts[pv_starts.year==year]
            pv_stops_year = pv_stops.shift(-1, freq='H')
            pv_stops_year = pv_stops_year[pv_stops_year.year==year]
            pv_stops_year = pv_stops_year.shift(1, freq='H')
            wind_starts_year = wind_starts[wind_starts.year==year]
            wind_stops_year = wind_stops.shift(-1, freq='H')
            wind_stops_year = wind_stops_year[wind_stops_year.year==year]
            wind_stops_year = wind_stops_year.shift(1, freq='H')
            
            for j, pv_start in enumerate(pv_starts_year):
                pv_stop = pv_stops_year[j]
                good_inds = (times>pv_start)&(times<pv_stop)&(np.invert(np.isnan(pv_tun_P)))&(np.invert(np.isnan(wind_gen)))
                good_pv_inds_all.extend([k+i*8760 for k, x in enumerate(good_inds) if x])
                good_pv_gen = pd.concat((good_pv_gen,pd.DataFrame([pv_gen[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
                good_pv_tun = pd.concat((good_pv_tun,pd.DataFrame([pv_tun_P[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
            for j, wind_start in enumerate(wind_starts_year):
                wind_stop = wind_stops_year[j]
                good_inds = (times>wind_start)&(times<wind_stop)&(np.invert(np.isnan(wind_tun_P)))
                if use_dir:
                    yaw_year = yaw_df.iloc[i*8760:(i+1)*8760]
                    wind_dir = yaw_year.loc[:,dir_label].values
                    good_inds = good_inds&(wind_dir<max_angle)&(wind_dir>min_angle)
                if use_status:
                    status_year = status_df.iloc[i*8760:(i+1)*8760]
                    status = status_year.loc[:,0].values
                    # status = np.hstack((status[1:],status[0]))
                    good_inds = good_inds&status
                good_wind_inds_all.extend([k+i*8760 for k, x in enumerate(good_inds) if x])
                good_wind_gen = pd.concat((good_wind_gen,pd.DataFrame([wind_gen[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
                good_wind_tun = pd.concat((good_wind_tun,pd.DataFrame([wind_tun_P[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
        
        
        # Calculate percent mismatch between modeling and actual data
        pct_pv_mismatch = (1-np.sum(good_pv_tun.values)/np.sum(good_pv_gen.values))*100
        pct_wind_mismatch = (1-np.sum(good_wind_tun.values)/np.sum(good_wind_gen.values))*100
        overshoots = {'pv': pct_pv_mismatch, 'wind': pct_wind_mismatch}

        if plot_val:
            
            # Generate power curves
            good_wind_speed = [wind_speed_all[i] for i in good_wind_inds_all]
            
            good_wind_dir = np.array([yaw_df.loc[:,dir_label].values[i] for i in good_wind_inds_all])
            good_wind_mismatch = np.array([yaw_df.loc[:,mismatch_label].values[i] for i in good_wind_inds_all])

            good_poa = [poa_all[i] for i in good_pv_inds_all]
            plt.subplot(1,2,1)
            plt.grid('on')
            plt.plot(good_poa,good_pv_tun.values,'.')
            plt.xlabel('Plane of array irradiance [W/m^2], 1 hour avg.')
            plt.ylabel('Active power [kW], 1 hour avg.')
            plt.subplot(1,2,2)
            plt.grid('on')
            plt.plot(wind_speed_all,wind_tun_all,'.',label='All measured data points',markersize=4)
            plt.plot([wind_speed_all[i] for i in np.where(status_df.iloc[:-1,0].values)[0]],
                        [wind_tun_all[i] for i in np.where(status_df.iloc[:-1,0].values)[0]],'.',label='Filtered data using status',markersize=4)
            plt.plot(good_wind_speed,good_wind_tun.values,'.',label='Filtered data using status & cleaning',markersize=4)
            plt.xlabel('Wind speed [m/s], 1 hour avg.')
            plt.ylabel('Active power [kW], 1 hour avg.')
            bin_starts = np.arange(0,20,.5)
            bin_ends = np.arange(.5,20.5,.5)
            ## Using manual bins for now, can uncomment this to get automatic binning
            # bin_interval = 0.25 # 0.25 m/s - Standard interval for binning wind power curves
            # bin_starts = np.arange(0,np.ceil(np.max(good_wind_speed)/bin_interval)*bin_interval,bin_interval)
            # bin_ends = bin_starts+bin_interval
            bin_speeds = []
            bin_powers = []
            for i, bin_start in enumerate(bin_starts):
                bin_end = bin_ends[i]
                bin_inds = np.where((good_wind_speed>bin_start)&(good_wind_speed<=bin_end))[0]
                bin_speeds.append(np.mean([good_wind_speed[i] for i in bin_inds]))
                bin_powers.append(np.mean(good_wind_tun.values[bin_inds]))
            plt.plot(bin_speeds,bin_powers,'-',label='New measured power curve',linewidth=2)
            plt.plot(hybrid.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds,\
                    hybrid.wind._system_model.Turbine.wind_turbine_powercurve_powerout,\
                    '-',label='Refactored power curve',linewidth=2)
            cd = Path(__file__).parent.absolute()
            wind_power_curve = cd / 'wind' / "iessGE15" / "NREL_Reference_1.5MW_Turbine_Site_Level.csv"
            curve_data = pd.read_csv(wind_power_curve)
            wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
            curve_power = curve_data['Power [kW]']
            plt.plot(wind_speed,curve_power,'-',label='Original power curve',linewidth=2)
            plt.legend()
            plt.show()
            bin_speeds.append(20)
            bin_powers.append(bin_powers[-1])

            plt.subplot(1,2,1)
            plt.grid('on')
            plt.plot(good_pv_tun.values,good_pv_gen.values,'.')
            plt.xlabel('Actual power [kW], 1 hour avg.')
            plt.ylabel('Modeled power [kW], 1 hour avg.')
            plt.plot([0,np.max(good_pv_gen.values)],[0,np.max(good_pv_gen.values)])
            plt.plot([0,np.max(good_pv_gen.values)],[0,np.max(good_pv_gen.values)*100/(100-pct_pv_mismatch)],
                        '--',label='PV model over-prediction: {:.2f}%'.format(pct_pv_mismatch))
            plt.legend()
            plt.subplot(1,2,2)
            plt.grid('on')
            plt.plot(good_wind_tun.values,good_wind_gen.values,'.')
            plt.xlabel('Actual power [kW], 1 hour avg.')
            plt.ylabel('Modeled power [kW], 1 hour avg.')
            plt.plot([0,np.max(good_wind_gen.values)],[0,np.max(good_wind_gen.values)])
            plt.plot([0,np.max(good_wind_gen.values)],[0,np.max(good_wind_gen.values)*100/(100-pct_wind_mismatch)],
                        '--',label='Wind model over-prediction: {:.2f}%'.format(pct_wind_mismatch))
            plt.legend()
            plt.show()

            # # Re-simulate with new loss coefficients
            # # tilt = copy.deepcopy(hybrid.pv._system_model.SystemDesign.tilt)
            # # azim = copy.deepcopy(hybrid.pv._system_model.SystemDesign.azimuth)
            # # tilt_adds = [0,-2.5]#np.linspace(-5,0,11)
            # # azim_adds = [0,-3]#np.linspace(-8,2,11)
            # # pv_gen_all_lols = []
            # # good_pv_gen_lols = []
            # # for k in range(len(tilt_adds)):
            # #     pv_gen_all_lols.append([])
            # #     good_pv_gen_lols.append([])
            # #     for _ in range(len(azim_adds)):
            # #         pv_gen_all_lols[k].append([])
            # #         good_pv_gen_lols[k].append(pd.DataFrame())
            # pv_gen_all = []
            # wind_gen_all = []
            # good_pv_gen = pd.DataFrame()
            # good_pv_tun = pd.DataFrame()
            # good_pv_inds_all = []
            # good_wind_gen = pd.DataFrame()
            # good_wind_tun = pd.DataFrame()
            # good_wind_inds_all = []
            # times_all = []

            # for i, year in enumerate(years):

            #     # Get good periods
            #     pv_starts_year = pv_starts[pv_starts.year==year]
            #     pv_stops_year = pv_stops.shift(-1, freq='H')
            #     pv_stops_year = pv_stops_year[pv_stops_year.year==year]
            #     pv_stops_year = pv_stops_year.shift(1, freq='H')
            #     wind_starts_year = wind_starts[wind_starts.year==year]
            #     wind_stops_year = wind_stops.shift(-1, freq='H')
            #     wind_stops_year = wind_stops_year[wind_stops_year.year==year]
            #     wind_stops_year = wind_stops_year.shift(1, freq='H')
                
            #     if year % 4 == 0:
            #         # Take out leap day
            #         times = pd.date_range(start=str(year)+'-01-01 00:30:00',periods=8784,freq='H')
            #         times = times[:1416].union(times[1440:])
            #     else:
            #         times = pd.date_range(start=str(year)+'-01-01 00:30:00',periods=8760,freq='H')
            #     times_all.extend(times)
                
            #     # Load actual generation data
            #     pv_tun_P = pv_tun_all[i*8760:(i+1)*8760]
            #     wind_tun_P = wind_tun_all[i*8760:(i+1)*8760]

            #     # Simulate generation for this specific year
            #     NewSolarRes = SolarResource(hybrid.site.lat,hybrid.site.lon,year,filepath=res_filepaths['pv'][i])
            #     NewWindRes = WindResource(hybrid.site.lat,hybrid.site.lon,year,hub_ht,filepath=res_filepaths['wind'][i])
            #     # Have to change pressure to sea level!
            #     for j in range(len(NewWindRes.data['data'])):
            #         NewWindRes.data['data'][j][1] = 1
            #     hybrid.pv._system_model.SolarResource.solar_resource_data = NewSolarRes.data
            #     hybrid.wind._system_model.Resource.wind_resource_data = NewWindRes.data

            #     hybrid.simulate_power(1)

            #     pv_gen = hybrid.pv.generation_profile
            #     pv_gen_all.extend(pv_gen)
            #     for j, pv_start in enumerate(pv_starts_year):
            #         pv_stop = pv_stops_year[j]
            #         good_inds = (times>pv_start)&(times<pv_stop)&(np.invert(np.isnan(pv_tun_P)))
            #         good_pv_inds_all.extend([k+i*8760 for k, x in enumerate(good_inds) if x])
            #         good_pv_gen = pd.concat((good_pv_gen,
            #             pd.DataFrame([pv_gen[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
            #         good_pv_tun = pd.concat((good_wind_tun,
            #             pd.DataFrame([pv_tun_P[i] for i in np.where(good_inds)[0]],index=times[good_inds])))            


            #     wind_gen = hybrid.wind.generation_profile
            #     wind_gen_all.extend(wind_gen)
            #     for j, wind_start in enumerate(wind_starts_year):
            #         wind_stop = wind_stops_year[j]
            #         good_inds = (times>wind_start)&(times<wind_stop)&(np.invert(np.isnan(wind_tun_P)))&(np.invert(np.isnan(wind_gen)))
            #         if use_dir:
            #             yaw_year = yaw_df.iloc[i*8760:(i+1)*8760]
            #             wind_dir = yaw_year.loc[:,dir_label].values
            #             good_inds = good_inds&(wind_dir<max_angle)&(wind_dir>min_angle)
            #         if use_status:
            #             status_year = status_df.iloc[i*8760:(i+1)*8760]
            #             status = status_year.loc[:,0].values
            #             # status = np.hstack((status[1:],status[0]))
            #             good_inds = good_inds&status
            #         good_wind_inds_all.extend([k+i*8760 for k, x in enumerate(good_inds) if x])
            #         good_wind_gen = pd.concat((good_wind_gen,pd.DataFrame([wind_gen[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
            #         good_wind_tun = pd.concat((good_wind_tun,pd.DataFrame([wind_tun_P[i] for i in np.where(good_inds)[0]],index=times[good_inds])))
            
            times_all = pd.DatetimeIndex(times_all)

            good_wind_dir = np.array([yaw_df.loc[:,dir_label].values[i] for i in good_wind_inds_all])
            good_wind_mismatch = np.array([yaw_df.loc[:,mismatch_label].values[i] for i in good_wind_inds_all])
            
            ax1 = plt.subplot(2,1,1)
            plt.grid('on')
            plt.plot(times_all,pv_gen_all,label='HOPP Modeled Generation')
            plt.plot(times_all,pv_tun_all,label='Actual Generation')
            plt.ylim([0,600])
            # plt.xlim(pd.DatetimeIndex(('2022-07-28','2022-08-14')))
            Ylim = ax1.get_ylim()
            for i, pv_start in enumerate(pv_starts):
                pv_stop = pv_stops[i]
                if i == 0:
                    label = 'Usable periods of "clean" data'
                else:
                    label = None
                good_period = patch.Rectangle([pv_start,Ylim[0]],pv_stop-pv_start,Ylim[1]-Ylim[0],color=[0,1,0],alpha=.5,label=label)
                ax1.add_patch(good_period)
            ax1.set_ylim(Ylim)
            plt.title('First Solar Array')
            plt.ylabel('Active Power [kW]')
            # plt.xlabel('Time')
            plt.legend() 
            ax2 = plt.subplot(2,1,2)
            ax2 = plt.gca()
            plt.grid('on')
            plt.plot(times_all,wind_gen_all,label='HOPP Modeled Output')
            plt.plot(times_all,wind_tun_all,label='ARIES Data')
            plt.ylim([0,1600])

            # consec_groups = np.split(good_wind_inds_all, np.where(np.diff(good_wind_inds_all) != 1)[0]+1)
            # for group in consec_groups:
            #     plt.plot(times_all[group],[wind_gen_all[i] for i in group],color=[1,0,0])
            #     plt.plot(times_all[group],[wind_tun_all[i] for i in group],color=[0,1,0])
            #     plt.plot(times_all[group],
            #                 np.subtract([wind_gen_all[i] for i in group],
            #                             [wind_tun_all[i] for i in group]),color=[0,0,1])
            
            # plt.plot([times_all[i] for i in good_wind_inds_all],[wind_gen_all[i] for i in good_wind_inds_all])
            # plt.plot([times_all[i] for i in good_wind_inds_all],[wind_tun_all[i] for i in good_wind_inds_all])
            # plt.plot([times_all[i] for i in good_wind_inds_all],np.subtract([wind_tun_all[i] for i in good_wind_inds_all],
                                                                            # [wind_gen_all[i] for i in good_wind_inds_all]),'s')
            Ylim = ax2.get_ylim()
            for i, wind_start in enumerate(wind_starts):
                wind_stop = wind_stops[i]
                if i == 0:
                    label = 'Usable periods of "clean" data'
                else:
                    label = None
                good_period = patch.Rectangle([wind_start,Ylim[0]],wind_stop-wind_start,Ylim[1]-Ylim[0],color=[0,1,0],alpha=.5,label=label)
                ax2.add_patch(good_period)
            ax2.set_ylim(Ylim)
            # plt.xlim(pd.DatetimeIndex(('2022-07-28','2022-08-14')))
            plt.title('GE Turbine')
            plt.ylabel('Active Power [kW]')
            # plt.xlabel('Time')
            plt.legend()
            plt.show()

            # Plot PV residuals
            pv_residuals = np.diff(np.vstack([pv_tun_all,pv_gen_all]),axis=0)[0][good_pv_inds_all]
            
            plt.subplot(2,2,1)
            # Time of day
            plt.grid('on')
            avgs = []
            stds = []
            hours = pd.date_range(start='00:30:00',periods=24,freq='H')
            num_hours = np.arange(0,24)
            for hour in num_hours:
                inds = np.where(good_pv_gen.index.hour==hour)[0]
                hour_residuals = [pv_residuals[i] for i in inds]
                avgs.append(np.mean(hour_residuals))
                stds.append(np.std(hour_residuals))
            plt.plot(num_hours,avgs,'k-')
            plt.plot(num_hours,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(num_hours,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.legend()
            plt.xlabel('Time of Day')
            plt.ylabel('Error (model - actual) [kW]')
            plt.subplot(2,2,2)
            # Power level
            plt.grid('on')
            avgs = []
            stds = []
            interval = 50
            bin_starts = np.arange(0,np.max(good_pv_gen.values),interval)
            for bin_start in bin_starts:
                inds = np.where((good_pv_gen.values>bin_start)&(good_pv_gen.values<=(bin_start+interval)))[0]
                power_residuals = [pv_residuals[i] for i in inds]
                avgs.append(np.mean(power_residuals))
                stds.append(np.std(power_residuals))
            avgs = np.divide(np.array(avgs),np.array(bin_starts)+interval/2)*100
            stds = np.divide(np.array(stds),np.array(bin_starts)+interval/2)*100
            plt.plot(bin_starts+interval/2,avgs,'k-')
            plt.plot(bin_starts+interval/2,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(bin_starts+interval/2,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Output level [kW]')
            plt.ylabel('Error (model - actual) [% of modeled output]')
            plt.subplot(2,2,3)
            # Month
            plt.grid('on')
            avgs = []
            stds = []
            months = pd.date_range(start='01/01/20',periods=12,freq='MS')
            num_months = np.arange(1,13)
            for month in num_months:
                inds = np.where(good_pv_gen.index.month==month)[0]
                month_residuals = [pv_residuals[i] for i in inds]
                avgs.append(np.mean(month_residuals))
                stds.append(np.std(month_residuals))
            plt.plot(num_months,avgs,'k-')
            plt.plot(num_months,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(num_months,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Month')
            plt.ylabel('Error (model - actual) [kW]')
            plt.subplot(2,2,4)
            # Year
            plt.grid('on')
            avgs = []
            stds = []
            for year in years:
                inds = np.where(good_pv_gen.index.year==year)[0]
                year_residuals = [pv_residuals[i] for i in inds]
                avgs.append(np.mean(year_residuals))
                stds.append(np.std(year_residuals))
            plt.plot(years,avgs,'k-')
            plt.plot(years,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(years,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Year')
            plt.ylabel('Error (model - actual) [kW]')
            plt.xticks(years)
            plt.show()

            # Plot Wind residuals
            wind_residuals = np.diff(np.vstack([wind_tun_all,wind_gen_all]),axis=0)[0][good_wind_inds_all]
            plt.clf()
            plt.subplot(2,2,1)
            # Time of day
            plt.grid('on')
            avgs = []
            stds = []
            hours = pd.date_range(start='00:30:00',periods=24,freq='H')
            num_hours = np.arange(0,24)
            for hour in num_hours:
                inds = np.where(good_wind_gen.index.hour==hour)[0]
                hour_residuals = [wind_residuals[i] for i in inds]
                avgs.append(np.mean(hour_residuals))
                stds.append(np.std(hour_residuals))
            plt.plot(num_hours,avgs,'k-')
            plt.plot(num_hours,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(num_hours,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Time of Day')
            plt.ylabel('Error (model - actual) [kW]')
            plt.subplot(2,2,2)
            # Power level
            plt.grid('on')
            avgs = []
            stds = []
            interval = 100
            # good_wind_gen_2 = [wind_gen_all[i] for i in good_wind_inds_all] # Don't know why I had to do this...
            bin_starts = np.arange(0,np.max(good_wind_gen.values),interval)
            for bin_start in bin_starts:
                inds = np.where((good_wind_gen>bin_start)&(good_wind_gen<=(bin_start+interval)))[0]
                power_residuals = [wind_residuals[i] for i in inds]
                avgs.append(np.mean(power_residuals))
                stds.append(np.std(power_residuals))
            avgs = np.divide(np.array(avgs),np.array(bin_starts)+interval/2)*100
            stds = np.divide(np.array(stds),np.array(bin_starts)+interval/2)*100
            plt.plot(bin_starts+interval/2,avgs,'k-')
            plt.plot(bin_starts+interval/2,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(bin_starts+interval/2,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Output level [kW]')
            plt.ylabel('Error (model - actual) [% of model]')
            # plt.subplot(2,3,3)
            # # Month
            # plt.grid('on')
            # avgs = []
            # stds = []
            # months = pd.date_range(start='01/01/20',periods=12,freq='MS')
            # num_months = np.arange(1,13)
            # for month in num_months:
            #     inds = np.where(good_wind_gen.index.month==month)[0]
            #     month_residuals = [wind_residuals[i] for i in inds]
            #     avgs.append(np.mean(month_residuals))
            #     stds.append(np.std(month_residuals))
            # plt.plot(num_months,avgs,'k-')
            # plt.plot(num_months,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            # plt.plot(num_months,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            # plt.xlabel('Month')
            # plt.ylabel('Error (model - actual) [kW]')
            # plt.subplot(2,3,4)
            # # Year
            # plt.grid('on')
            # avgs = []
            # stds = []
            # for year in years:
            #     inds = np.where(good_wind_gen.index.year==year)[0]
            #     year_residuals = [wind_residuals[i] for i in inds]
            #     avgs.append(np.mean(year_residuals))
            #     stds.append(np.std(year_residuals))
            # plt.plot(years,avgs,'k-')
            # plt.plot(years,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            # plt.plot(years,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            # plt.xlabel('Year')
            # plt.ylabel('Error (model - actual) [kW]')
            # plt.xticks(years)

            plt.subplot(2,2,3)
            # Wind Direction
            plt.grid('on')
            avgs = []
            stds = []
            dir_int = 10
            dir_bins = np.arange(0,360,dir_int)
            dir_bin_cts = []
            for dir_bin in dir_bins:
                inds = np.where((good_wind_dir>=dir_bin)&(good_wind_dir<(dir_bin+dir_int)))[0]
                dir_residuals = [wind_residuals[i] for i in inds]
                # avgs.append(np.mean(dir_residuals))
                avgs.append(np.mean(dir_residuals)/np.mean([good_wind_gen.iloc[i] for i in inds])*100)
                # stds.append(np.std(dir_residuals))
                stds.append(np.std(dir_residuals)/np.mean([good_wind_gen.iloc[i] for i in inds])*100)
                dir_bin_cts.append(len(inds))
            # plt.plot(dir_bins+dir_int/2,dir_bin_cts,'k-')
            plt.plot(dir_bins+dir_int/2,avgs,'k-')
            plt.plot(dir_bins+dir_int/2,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(dir_bins+dir_int/2,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Wind Direction [deg]')
            plt.ylabel('Error (model - actual) [% of model]')
            # plt.xticks(dir_bins)
            
            plt.subplot(2,2,4)
            # Yaw Mismatch
            plt.grid('on')
            avgs = []
            stds = []
            mismatch_int = 2
            mis_bins = np.arange(-29,29,mismatch_int)
            bin_cts = []
            for mis_bin in mis_bins:
                inds = np.where((good_wind_mismatch>=mis_bin)&(good_wind_mismatch<(mis_bin+mismatch_int)))[0]
                mis_residuals = [wind_residuals[i] for i in inds]
                avgs.append(np.mean(mis_residuals))
                stds.append(np.std(mis_residuals))
                bin_cts.append(len(inds))
            plt.plot(mis_bins+mismatch_int/2,avgs,'k-')
            plt.plot(mis_bins+mismatch_int/2,[avgs[i] + std for i, std in enumerate(stds)],'k--',)
            plt.plot(mis_bins+mismatch_int/2,[avgs[i] - std for i, std in enumerate(stds)],'k--',)
            plt.xlabel('Yaw Mismatch [deg]')
            plt.ylabel('Error (model - actual) [kW]')
            # plt.xticks(mis_bins)
            plt.show()

    return hybrid, overshoots