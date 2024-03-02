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
from hopp.simulation.technologies.fuel.simple_reactor import SimpleReactor
from hopp.utilities.log import hybrid_logger as logger


@define
class FuelConfig(BaseClass):
    """
    Configuration class for FuelPlant.

    Args:
        fuel_prod_kg_s: The fuel production rate in kg/s
        fuel_produced: The name of the fuel produced

    """
    fuel_prod_kg_s: float = field(default=1.0, validator=gt_zero)
    fuel_produced: str = field(default="hydrogen", validator=contains(["hydrogen","methanol"]))
    model_name: str = field(default="SimpleReactor", validator=contains(["SimpleReactor"]))
    simple_fin_config: Optional[dict] = field(default=SimpleFinanceConfig())
    model_input_file: Optional[str] = field(default=None)
    lca: Optional[dict] = field(default=None)
    
    
    def default():
        return FuelConfig(1.0)
    
default_config = FuelConfig.default()

@define
class FuelPlant(FlowSource):
    site: SiteInfo
    config: FuelConfig
    config_name: str = field(init=False, default="DefaultFuelPlant")
    simple_fin_config: SimpleFinanceConfig = field(default=None)

    def __attrs_post_init__(self):
        """
        FuelPlant

        Args:
            site: Site information
            config: Fuel plant configuration
        """
        
        if self.config is None:
            system_model = SimpleReactor(self.site,default_config,"fuel")
            financial_model = Singleowner.default('WindPowerSingleOwner')
        else:
            system_model = SimpleReactor(self.site,self.config,self.config.fuel_produced)
            if self.config.simple_fin_config:
                financial_model = SimpleFinance(self.config.simple_fin_config)
            else:
                financial_model = Singleowner.default('WindPowerSingleOwner')

        super().__init__("FuelPlant", self.site, system_model, financial_model)

        self.fuel_prod_kg_s = self.config.fuel_prod_kg_s
        self.fuel_produced = self.config.fuel_produced

    @property
    def fuel_prod_kg_s(self):
        return self._system_model.value("fuel_prod_kg_s")
    
    @fuel_prod_kg_s.setter
    def fuel_prod_kg_s(self, kg_s: float):
        self._system_model.value("fuel_prod_kg_s",kg_s)
    
    @property
    def fuel_produced(self):
        return self._system_model.value("fuel_produced")
    
    @fuel_produced.setter
    def fuel_produced(self, name: str):
        self._system_model.value("fuel_produced",name)

    @property
    def system_capacity_kg_s(self):
        return self._system_model.value("fuel_prod_kg_s")
    
    @system_capacity_kg_s.setter
    def system_capacity_kg_s(self, kg_s: float):
         self._system_model.value("fuel_prod_kg_s",kg_s)