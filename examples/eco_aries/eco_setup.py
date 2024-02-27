import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from yamlinclude import YamlIncludeConstructor

from hopp import ROOT_DIR
from hopp.simulation.technologies.sites import SiteInfo, oahu_site
from hopp.utilities import load_yaml
from hopp.simulation import HoppInterface
from hopp.utilities.keys import set_nrel_key_dot_env
from ORBIT.core.library import initialize_library
from greenheart.tools.eco.hybrid_system import run_simulation

dirname = os.path.dirname(__file__)
orbit_library_path = os.path.join(dirname, "input_files/")

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, 'floris/'))
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, 'turbines/'))

initialize_library(orbit_library_path)
set_nrel_key_dot_env()

def eco_setup(generate_ARIES_placeholders=False, plot_results=False):

    '''
    eco_setup()
    
    returns a HoppInterface instance with a baseline ECO (Energy Cluster Offshore) for validation with ARIES
    
    generate_ARIES_placeholders: bool (default False) to activate export of .csv placeholder ARIES signals
    plot_results: bool (default False) to activate plotting of simulated HOPP and placeholder ARIES signals
    '''

    sim_start = '2019-01-05 14:00:00.0'
    sim_end = '2019-01-06 14:00:00.0'        
    
    turbine_model = "iea_15MW"
    filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
    filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config.yaml")
    filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
    filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
    filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_aries_eco_baseline.yaml")

    hopp_results, elyzer_results, _ = run_simulation(filename_hopp_config, 
                                                    filename_eco_config, 
                                                    filename_turbine_config, 
                                                    filename_orbit_config, 
                                                    filename_floris_config, 
                                                    use_profast=True, 
                                                    incentive_option=1, 
                                                    plant_design_scenario=7, 
                                                    output_level=6,
                                                    post_processing=False,
                                                    skip_financials=True)
    hi = hopp_results['hopp_interface']
        
    if generate_ARIES_placeholders or plot_results:

        # Collect info from HOPP simulation to send to ARIES
        hybrid_plant = hi.system
        gen = hybrid_plant.generation_profile
        batt = hybrid_plant.battery.outputs
        wind_velocities = hi.system.wind._system_model.turb_velocities
        insol = hi.system.pv._system_model.Outputs.poa

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
        hybrid_gen = wind_gen+wave_gen+pv_gen+batt_gen

        # Double up the generation timepoints to make stepped plot with hopp_time2
        gen2_list = ["wind_gen2", "wave_gen2", "pv_gen2", "batt_gen2", "hybrid_gen2"]
        gen2_list = ["wind", "wave", "pv", "batt", "hybrid"]
        gen2_dict = {}
        for i, gen1 in enumerate([wind_gen, wave_gen, pv_gen, batt_gen, hybrid_gen]):
            gen2 = np.vstack([gen1,gen1])
            gen2 = np.reshape(np.transpose(gen2),8760*2)
            gen2_dict[gen2_list[i]] = gen2

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

            # Interpolate the HOPP generation and resources to 100 ms intervals
            hopp_time3 = pd.date_range('2019-01-01 00:30', periods=8760, freq='1 h')
            gen_frame = pd.DataFrame(np.transpose([wind_gen, wave_gen, pv_gen, batt_gen, hybrid_gen]),index=hopp_time3, columns=['wind','wave','solar', 'batt', 'elyzer'])
            slice_start = '2019-01-05 13:00'
            slice_end = '2019-01-06 15:00'
            gen_frame2 = gen_frame[slice_start:slice_end]
            gen_frame3 = gen_frame2.resample('30 min').interpolate('linear')
            gen_frame3 = gen_frame3[sim_start:sim_end]
            
            # Move the middle point to make the average the same
            for tech in gen_frame3.columns.values:
                values = gen_frame3[tech].values
                for i in range(int(len(values)/2)-1):
                    values[i*2+1] = (values[i*2+1]*4-values[i*2]-values[i*2+2])/2
                gen_frame3[tech] = values

            # Resample to 5 min
            gen_frame3 = gen_frame3.resample('5 min').interpolate('linear')

            # Superimpose some random noise
            for tech in gen_frame3.columns.values:
                values = gen_frame3[tech].values
                mean = np.mean(values)
                noise = np.random.standard_normal(len(values))*mean*.08 # based on evaluation of 5 min wind samples - std is 8% of mean
                gen_frame3[tech] = values+noise-np.mean(noise)

            # Resample to 100 ms
            gen_frame3 = gen_frame3.resample('100 ms').interpolate('cubic')

            # Put in placeholder battery command
            batt_placeholder_kw = -40000.
            gen_frame3["batt"] = np.full(gen_frame3['batt'].values.shape,batt_placeholder_kw)
            
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

    return hi, elyzer_results


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

if __name__ == '__main__':

    hi, _ = eco_setup(True, True)