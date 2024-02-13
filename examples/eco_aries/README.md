## Running HOPP-ARIES UDP communications framework for ECO

This is a framework for HOPP to simulate an ECO and send the results to 
a "real-time-balancer" which takes in live power generation data from a
real-time ARIES simulation. The balancer corrects HOPP's battery dispatch
signal to maintain a flat electrolyzer output in the ARIES simulation

How to run:

1. Install the 'socket' package in your conda environment
    ```
    conda activate hopp
    conda install socket
    ```
2. Get 3 terminals (or anaconda prompts on Windows) open,
activate conda environment and navigate to the directory in each:
    ```
    conda activate hopp
    cd hopp-nrel/examples/eco_aries
    ```
3. Run the three scripts, in this order, from each terminal.
Make sure to run all three within 60 seconds to avoid timeouts.
    Terminal 1:
    '''
    python realtime_balancer.py
    '''
    Terminal 2:
    '''
    python hopp_comms.py
    '''
    Terminal 3:
    '''
    python aries_comms.py
    '''
    
As it is, "aries_comms.py" puts out a placeholder signal that is not
a real ARIES simulation. This will be replaced by a true simulation.