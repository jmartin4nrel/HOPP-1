import socket
import numpy as np
import pandas as pd
import json
from hopp import ROOT_DIR
from examples.eco_aries.eco_setup import eco_setup, eco_modify

if __name__ == '__main__':

    bufferSize  = 4096

    # Set up hopp simulation
    hi, elyzer_results = eco_setup(True)
    hi.hopp.system.simulate_power(1)

    # Setup UDP send
    localIP     = "127.0.0.1"
    localPort   = 20001
    sendAddress   = (localIP, localPort)
    sendSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Setup UDP receive
    localIP     = "127.0.0.1"
    localPort   = 20004
    serverAddressPort   = (localIP, localPort)
    recvSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    recvSocket.bind(serverAddressPort)
    recvSocket.settimeout(60)

    # Set the time index to the start of the battery dispatch optimization 
    hopp_time = pd.date_range('2019-01-01', periods=8760, freq='1 h')
    start_timestep = hi.system.dispatch_options['limit_dispatch_idxs']['start_indx']
    hopp_timestep = hi.system.dispatch_options['limit_dispatch_idxs']['start_indx']
    end_timestep = hopp_timestep+24
    start_time = hopp_time[hopp_timestep]
    next_time = hopp_time[hopp_timestep+1]
    update_fraction = 0.8 # update dispatch optimization when ARIES has simulated 80% of the time period
    update_time = start_time+(next_time-start_time)*update_fraction
    new_SOC = hi.system.dispatch_options['limit_dispatch_idxs']['initial_soc']
    new_timestep = hopp_timestep-1

    # Set up a battery limit dict
    batt_lim_dict = {}
    batt_lim_dict['cap_kw'] = hi.system.battery.config.system_capacity_kw
    batt_lim_dict['cap_kwh'] = hi.system.battery.config.system_capacity_kwh
    batt_lim_dict['max_SOC'] = hi.system.battery.config.maximum_SOC
    batt_lim_dict['min_SOC'] = hi.system.battery.config.minimum_SOC

    # Send out HOPP-generated commands
    while hopp_timestep < end_timestep:

        if hopp_timestep != new_timestep:

            # Pull simulation results from target day
            hybrid_plant = hi.system
            gen = hybrid_plant.generation_profile
            batt = hybrid_plant.battery.outputs

            # Get generation of all generators as objects
            wind_gen = np.array(gen['wind'])
            wave_gen = np.array(gen['wave'])
            pv_gen = np.array(gen['pv'])
            batt_gen = np.array(gen['battery'])
            elyzer_load = np.array(elyzer_results['electrical_generation_timeseries'])
            hybrid_gen = wind_gen+wave_gen+pv_gen+batt_gen

            # Double up the generation timepoints to make stepped plot with hopp_time2
            gen_dict = {}
            gen_list = ["wind", "wave", "pv", "batt", "elyzer"," hybrid"]
            for i, gen1 in enumerate([wind_gen, wave_gen, pv_gen, batt_gen, elyzer_load, hybrid_gen]):
                gen_dict[gen_list[i]] = list(gen1[start_timestep:end_timestep])

            # Fill out the battery SOC time history
            batt_soc = np.array(batt.SOC)
            batt_soc[1:] = batt_soc[:-1]
            batt_soc[:(start_timestep+1)] = new_SOC
            batt_soc[(end_timestep+1):] = batt_soc[end_timestep+1]
            batt_soc = list(batt_soc[start_timestep:(end_timestep+1)])

            # Build command dict
            bess_kw = hi.system.generation_profile['battery'][hopp_timestep]
            elyzer_kw = hi.system.generation_profile['hybrid'][hopp_timestep]
            command_dict = {'bess_kw':bess_kw,'elyzer_kw':elyzer_kw}

            # Update timestep
            new_timestep = hopp_timestep

        # Give send json-encoded dict to ARIES
        whole_dict = {'commands':command_dict,
                    'gen':gen_dict,
                    'soc':batt_soc,
                    'batt_limits':batt_lim_dict}
        bytesToSend = str.encode(json.dumps(whole_dict))
        sendSocket.sendto(bytesToSend, sendAddress)

        # Receive ARIES time from balancer
        pair = recvSocket.recvfrom(bufferSize)
        aries_time = pd.DatetimeIndex([json.loads(pair[0])])[0]
        hopp_timestep = np.where(hopp_time == aries_time.floor('1h'))
        hopp_timestep = hopp_timestep[0][0]