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
from hopp.simulation.technologies.co2.simple_ngcc import SimpleNGCC, SimpleNGCC_Finance
from hopp.utilities.log import hybrid_logger as logger


@define
class CO2Config(BaseClass):
    """
    Configuration class for CO2Plant.

    Args:
        co2_kg_s: The CO2 production rate in kg/s
        capture_method: The name of the method used to capture CO2

    """
    co2_kg_s: float = field(default=1.0, validator=gt_zero)
    co2_model: str = field(default="SimpleNGCC", validator=contains(["SimpleNGCC"]))
    capture_model: str = field(default="AmineScrub", validator=contains(["AmineScrub","None"]))
    model_input_file: Optional[str] = field(default=None)
    lca: Optional[dict] = field(default=None)
    
    
    def default():
        return CO2Config(1.0)
    
default_config = CO2Config.default()

default_fin_config = {'life_yr':30,
                    'doll_yr':2020,
                    'capex':1e6,
                    'fopex_ann':1e3,
                    'vopex_kg':1e0,
                    'fcr':0.07}

@define
class CO2Plant(FlowSource):
    site: SiteInfo
    config: CO2Config
    config_name: str = field(init=False, default="DefaultCO2Plant")

    def __attrs_post_init__(self):
        """
        CO2Plant

        Args:
            site: Site information
            config: CO2 plant configuration
        """
        
        if self.config is None:
            system_model = SimpleNGCC(self.site,default_config,"ngcc_co2")
        else:
            if self.config.co2_model == 'SimpleNGCC':
                system_model = SimpleNGCC(self.site,self.config,"ngcc_co2")
        financial_model = Singleowner.default('WindPowerSingleOwner')

        super().__init__("CO2Plant", self.site, system_model, financial_model)

        self.co2_kg_s = self.config.co2_kg_s

    @property
    def co2_kg_s(self):
        return self._system_model.value("co2_kg_s")
    
    @co2_kg_s.setter
    def co2_kg_s(self, kg_s: float):
        self._system_model.value("co2_kg_s",kg_s)
    
    @property
    def system_capacity_kg_s(self):
        return self._system_model.value("co2_kg_s")
    
    @system_capacity_kg_s.setter
    def system_capacity_kg_s(self, kg_s: float):
         self._system_model.value("co2_kg_s",kg_s)