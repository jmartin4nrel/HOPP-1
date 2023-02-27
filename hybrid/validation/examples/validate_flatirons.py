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

# Period file to use for validation - put in data repository, <asset>/status/hybrid
periods = 'GE_FirstSolar_Periods_Recleaning.csv'

# Years of data to validate over
years = [2019,2020,2021,2022]

# Whether to plot results
plot_val = True

validate_asset(asset_path, config, periods, years, plot_val)