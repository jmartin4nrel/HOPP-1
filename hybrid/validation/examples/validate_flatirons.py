from pathlib import Path
import os

from hybrid.validation.validate_asset import validate_asset

cd = Path(__file__).parent.absolute()
hopp_base_dir = cd/'..'/'..'/'..'

# Relative path from HOPP repository to HOPP-Validation-Data repository
# MUST BE UPDATED FOR EACH USER, DEPENDING ON WHERE THEY PUT EACH
rel_path = hopp_base_dir/'..'/'..'/'jmartin4'
if not os.path.isdir(rel_path/'HOPP-Validation-Data'):
    raise NotADirectoryError('''HOPP-Validation-Data repository not found!
    May need to download repository and update path for your machine''')

# Asset to be validated - subfolder name in HOPP-Validation-Data repository
asset = 'Flatirons_IESS'
asset_path = rel_path/'HOPP-Validation-Data'/asset

# Configuration to be validated - matches subfolder in <asset>/system/hybrid/
# (Take out the "_pt#_of#" - this is used to combine different turbine types)
config = 'gefs_passive' # = GE turbine + FS (FirstSolar) PV array, no battery

# Manual status files - <asset>/status/wind/manual/Wind_Periods_<manual_fn>
#                   and <asset>/status/pv/manual/PV_Periods_<manual_fn>
# made from <asset>/status/hybrid/XXX_Periods_<manual_fn> using csv_splitter.py
manual_fn = 'Recleaning.csv'

# Limits - if single value, must match exactly; if 2-item list, [min, max]
# (Makes circular limits for degrees, e.g. [350, 10] will only be 20 deg window)
# TODO: Add key for names of resources/statuses/generated powers that can be called
limits = {  'pv':   {'manual':      True,
                     'gen_sim_kw':  [0,     500],
                     'gen_act_kw':  [0,     500]},
            'wind': {'manual':      True,
                     'status_bit':  True,
                     'speed_m_s':   [3,     25],
                     'dir_deg':     [243,   310],
                     'yaw':         [183,   10],
                     'yaw_mismatch':[-60,   60],
                     'gen_sim_kw':  [0,     2000],
                     'gen_act_kw':  [0,     2000]}}

# Run ID - change to overwrite previous sim results, leave the same to reload
run_id = 1

# Years of data to validate over
years = [2019,2020,2021,2022]

# Whether to plot results
plot_val = True

overshoots = validate_asset(asset_path, config, manual_fn, limits, run_id, years, plot_val, overwrite=True)