import multiprocessing
from eco_setup import eco_setup
from hopp_comms import hopp_comms
from aries_comms import aries_comms
from realtime_balancer import realtime_balancer
import time

if __name__ == '__main__':

    simulate_aries = True
    acceleration = 100
    num_inputs = 28
    initial_SOC = 50.0
    simulate_SOC = False

    eco_setup(True)

    hopp = multiprocessing.Process(target=hopp_comms)
    if simulate_aries:
        aries = multiprocessing.Process(target=aries_comms, args=(num_inputs, initial_SOC, acceleration))
    balancer = multiprocessing.Process(target=realtime_balancer, args=(simulate_aries,acceleration, simulate_SOC))

    balancer.start()
    time.sleep(5)
    if simulate_aries:
        aries.start()
        time.sleep(5)
    hopp.start()
    
    balancer.join()
    if simulate_aries:
        aries.join()
    hopp.join()
    