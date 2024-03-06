from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Set

from hopp.simulation.technologies.hydrogen.electrolyzer_plant import ElectrolyzerPlant
from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch
    
# if TYPE_CHECKING:
#     from hopp.simulation.technologies.hydrogen.electrolyzer_plant import ElectrolyzerPlant


class ElectrolyzerDispatch(PowerSourceDispatch):
    _system_model: ElectrolyzerPlant
    _financial_model: FinancialModelType
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: ElectrolyzerPlant,
                 financial_model: FinancialModelType,
                 block_set_name: str = 'electrolyzer'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

