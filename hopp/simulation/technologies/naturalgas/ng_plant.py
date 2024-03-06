from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

import PySAM.Singleowner as Singleowner
from attrs import define, field

from hopp.simulation.base import BaseClass
from hopp.type_dec import resource_file_converter
from hopp.utilities import load_yaml
from hopp.utilities.validators import gt_zero, contains
from hopp.simulation.technologies.flow_source import FlowSource
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp.simulation.technologies.financial import SimpleFinance, SimpleFinanceConfig
from hopp.simulation.technologies.naturalgas.simple_ng import SimpleNG, SimpleNG_Finance
from hopp.utilities.log import hybrid_logger as logger


@define
class NGConfig(BaseClass):
    """
    Configuration class for NGPlant.

    Args:
        ng_kg_s: The NG production rate in kg/s

    """
    ng_kg_s: float = field(default=1.0)
    ng_model: str = field(default="SimpleNG")
    simple_fin_config: Optional[dict] = field(default=None)
    model_input_file: Optional[str] = field(default=None)
    lca: Optional[dict] = field(default=None)
    
    
    def default():
        return NGConfig(1.0)
    
default_config = NGConfig.default()

@define
class NGPlant(FlowSource):
    site: SiteInfo
    config: NGConfig
    config_name: str = field(init=False, default="DefaultNGPlant")
    simple_fin_config: SimpleFinanceConfig = field(default=None)

    def __attrs_post_init__(self):
        """
        NGPlant

        Args:
            site: Site information
            config: NG plant configuration
        """
        
        if self.config is None:
            system_model = SimpleNG(self.site,default_config,"ng_co2")
        else:
            system_model = SimpleNG(self.site,self.config,"ng_co2")
        if self.config.simple_fin_config:
            financial_model = SimpleFinance(self.config.simple_fin_config)
            financial_model.system_capacity_kg_s = self.config.ng_kg_s
        else:
            financial_model = Singleowner.default('WindPowerSingleOwner')

        super().__init__("NGPlant", self.site, system_model, financial_model)

        self.ng_kg_s = self.config.ng_kg_s

    @property
    def ng_kg_s(self):
        return self._system_model.value("ng_kg_s")
    
    @ng_kg_s.setter
    def ng_kg_s(self, kg_s: float):
        self._system_model.value("ng_kg_s",kg_s)
    
    @property
    def system_capacity_kg_s(self):
        return self._system_model.value("ng_kg_s")
    
    @system_capacity_kg_s.setter
    def system_capacity_kg_s(self, kg_s: float):
         self._system_model.value("ng_kg_s",kg_s)

    @property
    def annual_mass_kg(self):
        return self._system_model.value("annual_mass_kg")
    
    @annual_mass_kg.setter
    def annual_mass_kg(self, kg: float):
        self._system_model.value("annual_mass_kg",kg)