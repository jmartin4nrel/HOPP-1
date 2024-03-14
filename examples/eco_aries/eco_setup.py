import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
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
        insol = hi.system.pv._system_model.Outputs.poa
        wind_speed = hi.system.wind._system_model.speeds
        wind_dir = hi.system.wind._system_model.wind_dirs
        wind_velocities = hi.system.wind._system_model.turb_velocities
        wind_x = hi.system.wind._system_model.fi.layout_x
        wind_y = hi.system.wind._system_model.fi.layout_y
        turb8_speed = wind_velocities[:,0,8]
        turb9_speed = wind_velocities[:,0,9]

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

        # Double up resources the same way
        insol2 = np.vstack([insol,insol])
        wind_speed2 = np.vstack([wind_speed,wind_speed])
        wind_dir2 = np.vstack([wind_dir,wind_dir])
        turb8_speed2 = np.vstack([turb8_speed,turb8_speed])
        turb9_speed2 = np.vstack([turb9_speed,turb9_speed])
        insol2 = np.reshape(np.transpose(insol2),8760*2)
        wind_speed2 = np.reshape(np.transpose(wind_speed2),8760*2)
        turb8_speed2 = np.reshape(np.transpose(turb8_speed2),8760*2)
        turb9_speed2 = np.reshape(np.transpose(turb9_speed2),8760*2)
        wind_dir2 = np.reshape(np.transpose(wind_dir2),8760*2)

        # Fill out the battery SOC time history
        batt_soc = np.array(batt.SOC)
        batt_soc[1:] = batt_soc[:-1]
        batt_soc[:111] = 50
        batt_soc[134:] = batt_soc[134]

        if plot_results:
            # Plot results
            plt.ioff()
            fig,ax=plt.subplots(3,1,sharex=True)
            fig.set_figwidth(12.0)
            fig.set_figheight(12.0)

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
            
            # plt.show()

        if generate_ARIES_placeholders:

            # Interpolate the HOPP generation and resources to 30 min intervals
            hopp_time3 = pd.date_range('2019-01-01 00:30', periods=8760, freq='1 h')
            aries_frame = pd.DataFrame(np.transpose([wind_gen, wave_gen, pv_gen, batt_gen, hybrid_gen, insol]),index=hopp_time3,
                                       columns=['wind','wave','solar', 'batt', 'elyzer', 'poa'])
            for col in range(wind_velocities.shape[2]):
                wind_vel = wind_velocities[:,0,col]
                aries_frame = aries_frame.join(pd.DataFrame(wind_vel,index=hopp_time3, columns=['wind_vel_{:02}'.format(col)]))
            aries_frame.to_csv(ROOT_DIR.parent / "examples" / "outputs" / "HOPP_output.csv")
            slice_start = '2019-01-05 13:00'
            slice_end = '2019-01-06 15:00'
            aries_frame = aries_frame[slice_start:slice_end]
            aries_frame = aries_frame.resample('30 min').interpolate('linear')
            aries_frame = aries_frame[sim_start:sim_end]
            
            # # Move the middle point to make the average the same as the HOPP input speed
            # for tech in aries_frame.columns.values:
            #     values = aries_frame[tech].values
            #     for i in range(int(len(values)/2)-1):
            #         values[i*2+1] = (values[i*2+1]*4-values[i*2]-values[i*2+2])/2
            #     aries_frame[tech] = values

            # Resample to 5 min
            aries_frame = aries_frame.resample('5 min').interpolate('linear')

            # Add in deviations
            wind_5min_devs = pd.read_csv(ROOT_DIR.parent/"examples"/"eco_aries"/"input_files"/"resources"/"wind_5min_dev.csv",header=None)
            for tech in aries_frame.columns.values:
                if 'wind_vel' in tech:
                    values = aries_frame[tech].values
                    values += np.squeeze(wind_5min_devs.values)
                    
            # Superimpose some random noise
            for tech in aries_frame.columns.values:
                if tech != 'poa':
                    values = aries_frame[tech].values
                    mean = np.mean(values)
                    # print(mean)
                    noise = np.random.standard_normal(len(values))*mean*.01 # Noise is 1% of the mean
                    noise = np.cumsum(noise)
                    noise = np.multiply(noise,np.divide(np.flip(np.divide(np.add(np.where(noise)[0],len(noise)),2)),len(noise)))
                    pos_noise = np.maximum(noise*0,values-noise)-values
                    # print(np.mean(values+pos_noise-np.mean(pos_noise)))
                    aries_frame[tech] = values+pos_noise-np.mean(pos_noise)

            # Resample to 1 s with spline fit
            aries_frame = aries_frame.resample('1 s').interpolate('cubic')

            # Make sure the signals stay above zero
            for tech in aries_frame.columns.values:
                values = aries_frame[tech].values
                aries_frame[tech] = np.maximum(values,0)

            # Calculate wind generation in each turbine
            A = 0.010612247
            B = 0.026944759
            C = -0.056860783
            D = -0.318909168
            cP_over_gen_times_vel_cubed = 35.8524092
            cutin = 2.8956 # Anything less ==> 0 MW
            cutoff = 10.66192 # Anything greater ==> 15 MW
            diameter = 242.24
            rho = 1.225
            pi = 3.14159
            ARIES_gen = 0
            HOPP_gen1 = 0
            num_turbs = wind_velocities.shape[2]
            frac_above_cutoff = 0
            for turb in range(num_turbs):
                vel = copy.deepcopy(aries_frame.iloc[:-1,turb+6].values)
                above_cutoffs = np.argwhere(vel>cutoff)
                frac_above_cutoff += len(above_cutoffs)/len(vel)/num_turbs
                vel[above_cutoffs] = cutoff
                vel[np.argwhere(vel<cutin)] = cutin
                gen = A*np.power(vel,3) + B*np.power(vel,2) + C*vel + D
                ARIES_gen += gen/3600
                speed = copy.deepcopy(wind_velocities[:,0,turb])
                speed[np.argwhere(speed>cutoff)] = cutoff
                speed[np.argwhere(speed<cutin)] = cutin
                # HOPPgen = A*np.power(speed,3) + B*np.power(speed,2) + C*speed + D
                cP_times_vel_cubed = (A*np.power(speed,3) + B*np.power(speed,2) + C*speed + D)*cP_over_gen_times_vel_cubed
                HOPPgen = rho/2*pi*(diameter/2)**2*cP_times_vel_cubed/1e6
                HOPP_gen1 += HOPPgen[110:134]
            ARIES_gen = np.sum(ARIES_gen)
            HOPP_gen1 = np.sum(HOPP_gen1)
            HOPP_gen2 = np.sum(wind_gen[110:134])/1000
            
            # Correct ARIES wind speeds
            iterations = 0
            ARIES_correction_factor = 1
            while iterations < 10 and ARIES_correction_factor > 0.0001:
                ARIES_correction_factor =  copy.deepcopy(HOPP_gen1/ARIES_gen-1)
                ARIES_gen = 0
                for turb in range(num_turbs):
                    vel = copy.deepcopy(aries_frame.iloc[:-1,turb+6].values)
                    below_cutoffs = np.argwhere(vel<cutoff)
                    idxs = copy.deepcopy(below_cutoffs)
                    distances = (cutoff-vel[below_cutoffs])/(cutoff-cutin)*2
                    vel[below_cutoffs] = vel[below_cutoffs]*(1+ARIES_correction_factor*distances)
                    new_vel_below = copy.deepcopy(vel[below_cutoffs])
                    vel[np.argwhere(vel<cutin)] = cutin
                    vel[np.argwhere(vel>cutoff)] = cutoff
                    gen = A*np.power(vel,3) + B*np.power(vel,2) + C*vel + D
                    ARIES_gen += gen[:-1]/3600
                    new_vel = aries_frame.iloc[:-1,turb+6].values
                    new_vel[idxs] = new_vel_below
                ARIES_gen = np.sum(ARIES_gen)
                iterations += 1

            print("Calculated wind generation from ARIES turbine speeds + IEA curve: {:.1f} MWh".format(ARIES_gen))
            print("Calculated wind generation from HOPP turbine speeds + IEA curve: {:.1f} MWh".format(HOPP_gen1))
            print("Calculated wind generation from plant-wide HOPP simulation: {:.1f} MWh".format(HOPP_gen2))

            # Find min and max wind turbines
            wind_velocities = aries_frame.iloc[:,6:].values
            max_wind_vel = np.max(wind_velocities,1)
            aries_frame = aries_frame.join(pd.DataFrame(np.transpose([max_wind_vel]),index=aries_frame.index.values, columns=['wind_vel_max']))
            min_wind_vel = np.min(wind_velocities,1)
            aries_frame = aries_frame.join(pd.DataFrame(np.transpose([min_wind_vel]),index=aries_frame.index.values, columns=['wind_vel_min']))

            # Put in placeholder battery command
            batt_placeholder_kw = -40000.
            aries_frame["batt"] = np.full(aries_frame['batt'].values.shape,batt_placeholder_kw)
            
            # Save to .csv
            aries_frame.to_csv(ROOT_DIR.parent / "examples" / "outputs" / "placeholder_ARIES.csv")

            hopp_time3 = aries_frame.index.values

            if plot_results:

                fig,ax=plt.subplots(3,1,sharex=True)
                fig.set_figwidth(12.0)
                fig.set_figheight(12.0)        

                ax[0].plot(hopp_time2,insol2,label = 'HOPP')
                ax[0].plot(hopp_time3,aries_frame['poa'],label = 'ARIES')
                ax[0].set_xlim(pd.DatetimeIndex((sim_start,sim_end)))
                ax[0].legend()
                ax[0].set_ylabel('POA insolation [W/m2]')

                ax[1].plot(hopp_time2,wind_speed2,'k-',linewidth=3,label = 'Original HOPP wind speed')
                ax[1].plot(hopp_time2,turb8_speed2,'c--',label = 'Turbine 8 HOPP eff. rotor vel.')
                ax[1].plot(hopp_time2,turb9_speed2,'r:',label = 'Turbine 9 HOPP eff. rotor vel.')
                ax[1].plot(hopp_time3,aries_frame['wind_vel_08'],'c-',label = 'Turbine 8 ARIES wind speed')
                ax[1].plot(hopp_time3,aries_frame['wind_vel_09'],'r-',label = 'Turbine 9 ARIES wind speed')
                # ax[1].plot(hopp_time3,aries_frame['wind_vel_max'],'--',label = 'Max speed of all ARIES turbines')
                # ax[1].plot(hopp_time3,aries_frame['wind_vel_min'],'--',label = 'Min speed of all ARIES turbines')
                ax[1].legend(ncol=2)
                ax[1].set_ylabel('Wind Speed [m/s]')
                ax[1].set_ylim([0,15])

                ax[2].plot(hopp_time2,wind_dir2,)
                ax[2].set_ylabel('Wind Direction [deg]')
                ax[2].set_ylim([80,140])

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