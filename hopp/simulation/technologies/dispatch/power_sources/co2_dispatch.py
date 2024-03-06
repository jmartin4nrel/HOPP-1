from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Set

from hopp.simulation.technologies.co2.co2_plant import CO2Plant
from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch
    

class Co2Dispatch(PowerSourceDispatch):
    _system_model: CO2Plant
    _financial_model: FinancialModelType
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: CO2Plant,
                 financial_model: FinancialModelType,
                 block_set_name: str = 'co2'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

