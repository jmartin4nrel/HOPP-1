import multiprocessing
from eco_setup import eco_setup
from hopp_comms import hopp_comms
from aries_comms import aries_comms
from realtime_balancer import realtime_balancer
import time

if __name__ == '__main__':

    simulate_aries = True

    eco_setup(True)

    hopp = multiprocessing.Process(target=hopp_comms)
    if simulate_aries:
        aries = multiprocessing.Process(target=aries_comms)
    balancer = multiprocessing.Process(target=realtime_balancer, args=(simulate_aries,))

    hopp.start()
    time.sleep(1)
    balancer.start()
    time.sleep(1)
    if simulate_aries:
        aries.start()

    hopp.join()
    balancer.join()
    if simulate_aries:
        aries.join()