from typing import Union

import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.technologies.financial.simple_financial_model import SimpleFinance, SimpleFinanceConfig

FinancialModelType = Union[Singleowner.Singleowner, CustomFinancialModel]
