import multiprocessing
from hopp_comms import hopp_comms
from aries_comms import aries_comms
from realtime_balancer import realtime_balancer
import time

if __name__ == '__main__':

    hopp = multiprocessing.Process(target=hopp_comms)
    aries = multiprocessing.Process(target=aries_comms)
    balancer = multiprocessing.Process(target=realtime_balancer)

    hopp.start()
    time.sleep(10)
    balancer.start()
    time.sleep(10)
    aries.start()

    hopp.join()
    balancer.join()
    aries.join()