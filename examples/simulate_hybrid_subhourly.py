from pathlib import Path
import sys
sys.path.append('/home/gstarke/Research_Programs/HOPP/HOPP/')
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
from hybrid.keys import set_nrel_key_dot_env
# Set API key
set_nrel_key_dot_env()

examples_dir = Path(__file__).parent.absolute()

solar_size_mw = 50
wind_size_mw = 50
battery_capacity_mw = 20
interconnection_size_mw = 50

technologies = {
    'pv': {
        'system_capacity_kw': solar_size_mw * 1000,
    },
    'wind': {
        'num_turbines': 25,
        'turbine_rating_kw': int(wind_size_mw * 1000 / 25)
    },
    'battery': {
        'system_capacity_kwh': battery_capacity_mw * 1000,
        'system_capacity_kw': battery_capacity_mw * 4 * 1000
    }
}

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
# prices_file = examples_dir.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
prices_file = examples_dir.parent / "timescale_test_files" / "pricing_data_test.csv"
wind_file = examples_dir.parent / "timescale_test_files" / "wind_test.srw"
solar_file = examples_dir.parent / "timescale_test_files" / "35.2018863_-101.945027_psmv3_60_2012.csv"
site = SiteInfo(flatirons_site, wind_resource_file=wind_file, solar_resource_file=solar_file,
                grid_resource_file=prices_file)
print('n_timesteps', site.n_timesteps)
print('n periods per day', site.n_periods_per_day)
print('n periods per day', site.interval)
# jkjkjk
# site.resample_data('30T')     

print('n_timesteps', site.n_timesteps)
print('n periods per day', site.n_periods_per_day)
print('n periods per day', site.interval)

dispatch_options = {"n_look_ahead_periods": site.n_periods_per_day*2, \
                        "n_roll_periods": site.n_periods_per_day}

# Create base model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000,\
                dispatch_options=dispatch_options)

hybrid_plant.pv.dc_degradation = (0,)             # year over year degradation
hybrid_plant.wind.wake_model = 3                # constant wake loss, layout-independent
hybrid_plant.wind.value("wake_int_loss", 1)     # percent wake loss

hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)

# prices_file are unitless dispatch factors, so add $/kwh here
hybrid_plant.ppa_price = 0.04

# use single year for now, multiple years with battery not implemented yet
hybrid_plant.simulate(project_life=1)

print("output after losses over gross output",
      hybrid_plant.wind.value("annual_energy") / hybrid_plant.wind.value("annual_gross_energy"))

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values
revs = hybrid_plant.total_revenues
print(annual_energies)
print(npvs)
print(revs)


file = 'figures/'
tag = 'simple2_'
#plot_battery_dispatch_error(hybrid_plant, plot_filename=file+tag+'battery_dispatch_error.png')
'''
for d in range(0, 360, 5):
    plot_battery_output(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_battery_gen.png')
    plot_generation_profile(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_system_gen.png')
'''
# plot_battery_dispatch_error(hybrid_plant)
# plot_battery_output(hybrid_plant)
plot_generation_profile(hybrid_plant)
#plot_battery_dispatch_error(hybrid_plant, plot_filename=tag+'battery_dispatch_error.png')
