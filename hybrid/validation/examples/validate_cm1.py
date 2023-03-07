from pathlib import Path
import os

from hybrid.validation.validate_asset import validate_asset

# EACH NEW USER MUST GET ACCESS TO THIS DIRECTORY - CONFIDENTIAL DATA!
y_drive_path = Path('Y:/5000/Projects/HOPP Validation/Asset Data')
if not os.path.isdir(y_drive_path):
    raise NotADirectoryError('''Asset data repository not found!
    May need access to Y drive directory - contact project PI''')

# Asset to be validated - subfolder name in asset data repository
asset = 'Copper_Mountain_I'
asset_path = y_drive_path/asset

# Configuration to be validated - matches subfolder in <asset>/system/hybrid/
# (Take out the "_pt#_of#" - this is used to combine different turbine types)
config = 'cm1' # = GE turbine + FS (FirstSolar) PV array, no battery

# Manual status files - <asset>/status/wind/manual/Wind_Periods_<manual_fn>
#                   and <asset>/status/pv/manual/PV_Periods_<manual_fn>
# made from <asset>/status/hybrid/XXX_Periods_<manual_fn> using csv_splitter.py
manual_fn = 'Uncleaned.csv'

# Limits - if single value, must match exactly; if 2-item list, [min, max]
# (Makes circular limits for degrees, e.g. [350, 10] will only be 20 deg window)
# TODO: Add key for names of resources/statuses/generated powers that can be called
limits = {  'pv':   {'manual':      True,
                     'gen_sim_kw':  [0,     50000],
                     'gen_act_kw':  [0,     50000]}}

# Run ID - change to overwrite previous sim results, leave the same to reload
run_id = 0

# Years of data to validate over
years = [2011]

# Whether to plot results
plot_val = True

overshoots = validate_asset(asset_path, config, manual_fn, limits, run_id, years, plot_val)