import json
from pathlib import Path

site = {
    "lat": 39.91,
    "lon": -105.22,
    "elev": 1835,
    "year": 2013,
    "tz": -7,
    'site_boundaries': {
        'verts': [[3.0599999999976717, 288.87000000011176],
                    [0.0, 1084.0300000002608],
                    [1784.0499999999884, 1084.2400000002235],
                    [1794.0900000000256, 999.6399999996647],
                    [1494.3400000000256, 950.9699999997392],
                    [712.640000000014, 262.79999999981374],
                    [1216.9800000000396, 272.3600000003353],
                    [1217.7600000000093, 151.62000000011176],
                    [708.140000000014, 0.0]],
        'verts_simple': [[3.0599999999976717, 288.87000000011176],
                        [0.0, 1084.0300000002608],
                        [1784.0499999999884, 1084.2400000002235],
                        [1794.0900000000256, 999.6399999996647],
                        [1216.9800000000396, 272.3600000003353],
                        [1217.7600000000093, 151.62000000011176],
                        [708.140000000014, 0.0]]
    },
    'urdb_label': "5ca4d1175457a39b23b3d45e"
}

technologies = {'pv': {
                    'system_capacity_kw': 430
                },
                'wind': {
                    'num_turbines': 1,
                    'turbine_rating_kw': 1500,
                    'hub_height': 80,
                    'rotor_diameter': 77
                },
                'interconnect_kw': 2000}

system_constants = {'pv': {
                        'azimuth': 180,
                        'tilt': 25,
                        'dc_ac_ratio': 1,
                        'losses': 2.964369166,
                        'array_type': 0
                        },
                    'wind': {
                        'avail_bop_loss': 0,
                        'avail_grid_loss': 0,
                        'avail_turb_loss': 0,
                        'elec_eff_loss': 0,
                        'elec_parasitic_loss': 0,
                        'env_degrad_loss': 0,
                        'env_env_loss': 0,
                        'env_icing_loss': 0,
                        'ops_env_loss': 0,
                        'ops_grid_loss': 0,
                        'ops_load_loss': 0,
                        'turb_generic_loss': 0,
                        'turb_hysteresis_loss': 0,
                        'turb_perf_loss': 0,
                        'turb_specific_loss': 8.289389901,
                        'wake_ext_loss': 0
                        }
                    }


cd = Path(__file__).parent.absolute()
with open(cd/'system_constants.json', 'w') as w:
    json.dump(system_constants, w)