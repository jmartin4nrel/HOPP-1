from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Set

from hopp.simulation.technologies.grid_sales import GridSales
from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch
    

class Grid_salesDispatch(PowerSourceDispatch):
    _system_model: GridSales
    _financial_model: FinancialModelType
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: GridSales,
                 financial_model: FinancialModelType,
                 block_set_name: str = 'grid_sales'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

