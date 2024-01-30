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
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.simulation.technologies.fuel.simple_reactor import SimpleReactor, SimpleReactorFinance
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
    model_input_file: Optional[str] = field(default=None)
    
    
    def default():
        return FuelConfig(1.0)
    
default_config = FuelConfig.default()

default_fin_config = {'life_yr':30,
                    'doll_yr':2020,
                    'capex':1e6,
                    'fopex_ann':1e3,
                    'vopex_kg':1e0,
                    'fcr':0.07}

@define
class FuelPlant(FlowSource):
    site: SiteInfo
    config: FuelConfig
    config_name: str = field(init=False, default="DefaultFuelPlant")

    def __attrs_post_init__(self):
        """
        FuelPlant

        Args:
            site: Site information
            config: Fuel plant configuration
        """
        
        if self.config is None:
            system_model = SimpleReactor(self.site,default_config,"default fuel plant")
        else:
            system_model = SimpleReactor(self.site,self.config,self.config.fuel_produced+" plant")
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