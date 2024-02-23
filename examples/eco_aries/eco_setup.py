from hopp import ROOT_DIR
from hopp.simulation.technologies.sites import SiteInfo, oahu_site
from hopp.utilities import load_yaml
from hopp.simulation import HoppInterface
from hopp.utilities.keys import set_nrel_key_dot_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from hopp.utilities.keys import set_developer_nrel_gov_key

# # yaml imports
import yaml
from yamlinclude import YamlIncludeConstructor
from pathlib import Path

PATH = Path(__file__).parent
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / './input/floris/')
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / './input/turbines/')

# ORBIT imports
from ORBIT.core.library import initialize_library
initialize_library(os.path.join(os.getcwd(), "./../eco/05-offshore-h2/input/"))

# HOPP imports
from greenheart.tools.eco.hybrid_system import run_simulation

# Set API key manually if not using the .env
global NREL_API_KEY
NREL_API_KEY = os.getenv("NREL_API_KEY") # Set this key manually here if you are not setting it using the .env
set_developer_nrel_gov_key(NREL_API_KEY)  

def eco_setup(generate_ARIES_placeholders=False, plot_results=False):

    '''
    eco_setup()
    
    returns a HoppInterface instance with a baseline ECO (Energy Cluster Offshore) for validation with ARIES
    
    generate_ARIES_placeholders: bool (default False) to activate export of .csv placeholder ARIES signals
    plot_results: bool (default False) to activate plotting of simulated HOPP and placeholder ARIES signals
    '''

    sim_start = '2019-01-05 14:00:00.0'
    sim_end = '2019-01-06 14:00:00.0'        
    
    # Set the desired load schedule to control the battery dispatch
    DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR.parent / "examples" / "inputs" / "resource_files" / "eco" / "oahu_N_19_18_loop_solar_resource.csv"
    DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR.parent / "examples" / "inputs" / "resource_files" / "eco" / "oahu_N_19_18_loop_wind_resource_srw.srw"
    DEFAULT_WAVE_RESOURCE_FILE = ROOT_DIR.parent / "examples" / "inputs" / "resource_files" / "eco" / "oahu_N_19_18_loop_wave_resource_3hr.csv"
    DEFAULT_PRICE_FILE = ROOT_DIR.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
    elzyer_load_kw = float(360 * 1000)
    DEFAULT_LOAD = elzyer_load_kw*np.ones((8760))/1000
    site = SiteInfo(
            oahu_site,
            solar_resource_file=DEFAULT_SOLAR_RESOURCE_FILE,
            wind_resource_file=DEFAULT_WIND_RESOURCE_FILE,
            wave_resource_file=DEFAULT_WAVE_RESOURCE_FILE,
            grid_resource_file=DEFAULT_PRICE_FILE,
            desired_schedule=DEFAULT_LOAD,
            solar=True,
            wind=True,
            wave=True
        )
    # Create the HOPP Model
    CONFIG_FILE = ROOT_DIR.parent / "examples" / "inputs" / "09-eco_aries.yaml"
    hopp_config = load_yaml(CONFIG_FILE)
    hopp_config["site"] = site
    hi = HoppInterface(hopp_config)

    # Enter turbine power curve and eliminate losses
    wind_power_curve = ROOT_DIR.parent / "examples" / "inputs" / "resource_files" / "eco" / "iea15mw_power_curve.csv"
    curve_data = pd.read_csv(wind_power_curve)
    wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
    curve_power = [i*1000 for i in curve_data['Power [MW]']]
    hi.system.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
    hi.system.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power 
    all_losses = 0.0
    loss_list = ["avail_bop_loss","avail_grid_loss","avail_turb_loss","elec_eff_loss","elec_parasitic_loss","env_degrad_loss", "env_env_loss", "env_icing_loss", "ops_env_loss", "ops_grid_loss", "ops_load_loss", "turb_generic_loss", "turb_hysteresis_loss", "turb_perf_loss", "turb_specific_loss", "wake_ext_loss"]
    for loss in loss_list:
        getattr(hi.system, 'wind').value(loss,all_losses)

    # Add Wave Cost Model Inputs
    cost_model_inputs = {
        'reference_model_num':3,
        'water_depth': 100,
        'distance_to_shore': 80,
        'number_rows': 10,
        'device_spacing':600,
        'row_spacing': 600,
        'cable_system_overbuild': 20
    }
    hi.system.wave.create_mhk_cost_calculator(cost_model_inputs)

    # # Create HOPP model - new
    # turbine_model="osw_18MW"
    # filename_orbit_config= "./../eco/05-offshore-h2/input/plant/orbit-config-"+turbine_model+".yaml"
    # filename_turbine_config = "./../eco/05-offshore-h2/input/turbines/"+turbine_model+".yaml"
    # filename_floris_config = "./../eco/05-offshore-h2/input/floris/floris_input_osw_18MW.yaml"
    # filename_hopp_config = "./../eco/05-offshore-h2/input/plant/hopp_config.yaml"
    # filename_eco_config = "./../eco/05-offshore-h2/input/plant/eco_config.yaml"
    # hopp_results, electrolyzer_physics_results, remaining_power_profile = run_simulation(\
    #     filename_hopp_config, filename_eco_config, filename_turbine_config, filename_orbit_config, filename_floris_config,
    #     verbose=False, show_plots=False, save_plots=False, use_profast=True, incentive_option=1, plant_design_scenario=1,
    #     output_level=6, post_processing=False)
    # hi = hopp_results{'hopp_interface'}
    
    if generate_ARIES_placeholders or plot_results:

        hi.hopp.system.simulate_power(1)
        hybrid_plant = hi.system
        gen = hybrid_plant.generation_profile
        batt = hybrid_plant.battery.outputs

        # Make time series - "hopp_time" with one point per hour, "hopp_time2" with two points per hour
        hopp_time = pd.date_range('2019-01-01', periods=8761, freq='1 h')
        hopp_time2 = np.vstack([hopp_time,hopp_time])
        hopp_time2 = np.reshape(np.transpose(hopp_time2),8761*2)
        hopp_time2 = hopp_time2[1:-1]
        hopp_time = hopp_time[:-1]

        # Get generation of all generators as objects
        wind_gen = np.array(gen['wind'])
        wave_gen = np.array(gen['wave'])
        pv_gen = np.array(gen['pv'])
        batt_gen = np.array(gen['battery'])
        hybrid_gen = np.array(gen['hybrid'])

        # Double up the generation timepoints to make stepped plot with hopp_time2
        gen2_list = ["wind_gen2", "wave_gen2", "pv_gen2", "batt_gen2", "hybrid_gen2"]
        gen2_list = ["wind", "wave", "pv", "batt", "hybrid"]
        gen2_dict = {}
        for i, gen1 in enumerate([wind_gen, wave_gen, pv_gen, batt_gen, hybrid_gen]):
            gen2 = np.vstack([gen1,gen1])
            gen2 = np.reshape(np.transpose(gen2),8760*2)
            gen2_dict[gen2_list[i]] = gen2
            # exec(gen2_list[i]+" = gen2")

        # Fill out the battery SOC time history
        batt_soc = np.array(batt.SOC)
        batt_soc[1:] = batt_soc[:-1]
        batt_soc[:111] = 50
        batt_soc[134:] = batt_soc[134]

        if plot_results:
            # Plot results
            plt.ioff()
            fig,ax=plt.subplots(3,1,sharex=True)
            fig.set_figwidth(8.0)
            fig.set_figheight(9.0)

            ax[0].plot(hopp_time2,gen2_dict['wave']/1000,label="Wave Generation")
            ax[0].plot(hopp_time2,gen2_dict['pv']/1000,label="Solar Generation")
            ax[0].legend()
            ax[0].set_ylabel('Generation [MW]')

            ax[1].plot(hopp_time2,gen2_dict['wave']/1000,label="Wave Generation")
            ax[1].plot(hopp_time2,gen2_dict['pv']/1000,label="Solar Generation")
            ax[1].plot(hopp_time2,gen2_dict['wind']/1000,label="Wind Generation")
            ax[1].plot(hopp_time2,gen2_dict['batt']/1000,label="Battery Generation")
            ax[1].plot(hopp_time2,gen2_dict['hybrid']/1000,label="Output to Electrolyzer")
            ax[1].legend(ncol=3)
            ax[1].set_ylabel('Generation [MW]')
            ax[1].set_ylim([-120,550])


            ax[2].plot(hopp_time,batt_soc,'k-')
            ax[2].set_ylabel('Battery SOC [%]')
            ax[2].set_ylim([0,100])

            plt.xlim(pd.DatetimeIndex((sim_start,sim_end)))
            
            plt.show()

        if generate_ARIES_placeholders:

            # Interpolate the HOPP generation to 100 ms intervals
            hopp_time3 = pd.date_range('2019-01-01 00:30', periods=8760, freq='1 h')
            gen_frame = pd.DataFrame(np.transpose([wind_gen, wave_gen, pv_gen, batt_gen, hybrid_gen]),index=hopp_time3, columns=['wind','wave','solar', 'batt', 'elyzer'])
            slice_start = '2019-01-05 13:00'
            slice_end = '2019-01-06 15:00'
            gen_frame2 = gen_frame[slice_start:slice_end]
            gen_frame3 = gen_frame2.resample('100 ms').interpolate('cubic')
            gen_frame3 = gen_frame3[sim_start:sim_end]

            # Superimpose some random noise
            for tech in gen_frame3.columns.values:
                values = gen_frame3[tech].values
                mean = np.mean(values)
                gen_frame3[tech] = values+np.random.standard_normal(len(values))*mean/100

            # Put in placeholder battery command
            batt_placeholder_kw = -40000.
            gen_frame3["batt"] = np.full(values.shape,batt_placeholder_kw)
            
            # Save to .csv
            gen_frame3.to_csv(ROOT_DIR.parent / "examples" / "outputs" / "placeholder_ARIES.csv")

            hopp_time3 = gen_frame3.index.values
            wind_gen3 = gen_frame3['wind'].values

            if plot_results:

                fig,ax=plt.subplots(2,1)
                fig.set_figwidth(8.0)
                fig.set_figheight(6.0)

                ax[0].plot(hopp_time3,wind_gen3/1000,label = '"ARIES modeled" wind (placeholder)')
                ax[0].plot(hopp_time2,gen2_dict['wind']/1000,label = 'HOPP modeled wind')
                ax[0].set_xlim(pd.DatetimeIndex((sim_start,sim_end)))
                ax[0].legend()
                ax[0].set_ylabel('Generation [MW]')

                zoom_start = '2019-01-05 14:30'
                zoom_end = '2019-01-05 14:31'
                ax[1].plot(hopp_time3,wind_gen3/1000,label = '"ARIES modeled" wind (placeholder)')
                ax[1].plot(hopp_time2,gen2_dict['wind']/1000,label = 'HOPP modeled wind')
                ax[1].set_xlim(pd.DatetimeIndex((zoom_start,zoom_end)))
                ax[1].legend()
                ax[1].set_ylabel('Generation [MW]')
                ax[1].set_ylim([340,380])

                plt.show()

    return hi


def eco_modify(hi, timestep, results):

    '''
    eco_modify

    modifies an existing ECO (Energy Cluster Offshore) based on ARIES validation results

    hi: HoppInterface instance previously created by eco_setup
    timestep: int hour of year over which ARIES has simulated 80% and results are being fed back to HOPP
    results: dict ARIES feedback to HOPP for this timestep, projected out to full hour from 80% complete simulated hour 
        {'gen':     {'wind':    float wind generation in kW,
                    'wave':     float wave generation in kW,
                    'solar':    float solar generation in kW,},
         'load':    {'elyzer':  float electrolyzer load in kW,
                    'periph':   float peripheral load in kW},
         'SOC'      float battery SOC in % at end of the hour}
    '''

    return hi