from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel
from attrs import define, field
from typing import Optional, TYPE_CHECKING
from hopp.tools.analysis.bos.bos_lookup import BOSLookup
from collections.abc import Iterable 
import numpy as np

hybrid_output_mw = 140
wind_cap = 0.45
solar_cap = 0.22
for wind_pct in np.arange(1,100,1):
    wind_mw = hybrid_output_mw*wind_pct/100/wind_cap
    pv_mw = hybrid_output_mw*(100-wind_pct)/100/solar_cap
    interconnect_mw = wind_mw+pv_mw
    bos_lookup = BOSLookup()
    _, _, solar_project_cost, _ = bos_lookup.calculate_bos_costs(0,
                                                                pv_mw,
                                                                interconnect_mw,
                                                                scenario='simple financial')
    _, _, wind_project_cost, _ = bos_lookup.calculate_bos_costs(wind_mw,
                                                                0,
                                                                interconnect_mw,
                                                                scenario='simple financial')
    _, _, total_project_cost, _ = bos_lookup.calculate_bos_costs(wind_mw,
                                                                pv_mw,
                                                                interconnect_mw,
                                                                scenario='simple financial')
    bos_savings = total_project_cost/(solar_project_cost+wind_project_cost)
    if isinstance(bos_savings, Iterable):
        bos_savings = bos_savings[0]
    pct_saved = (1-bos_savings)*100
    print(solar_project_cost, wind_project_cost, pct_saved)
    