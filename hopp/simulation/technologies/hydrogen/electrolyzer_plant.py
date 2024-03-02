from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

import PySAM.Singleowner as Singleowner
from attrs import define, field

from hopp.simulation.base import BaseClass
from hopp.type_dec import resource_file_converter
from hopp.utilities import load_yaml
from hopp.utilities.validators import gt_zero, contains
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp.simulation.technologies.financial import SimpleFinance, SimpleFinanceConfig
from hopp.simulation.technologies.hydrogen.simple_electrolyzer import SimpleElectrolyzer
from hopp.utilities.log import hybrid_logger as logger


@define
class ElectrolyzerConfig(BaseClass):
    """
    Configuration class for ElectrolyzerPlant.

    Args:
        capacity_kw: The capacity in kw

    """
    capacity_kw: float = field(default=1.0, validator=gt_zero)
    model_name: str = field(default="SimpleElectrolyzer", validator=contains(["SimpleElectrolyzer"]))
    generation_profile: list = field(default=[0.0]*8760)
    kwh_kg_h2: float = field(default=54.66)
    kg_h2o_kg_h2: float = field(default=14.2884)
    input_streams_kg_s: Optional[dict] = field(default={'water':[0.0]*8760})
    output_streams_kg_s: Optional[dict] = field(default={'hydrogen':[0.0]*8760})
    simple_fin_config: Optional[dict] = field(default=(SimpleFinanceConfig()))
    model_input_file: Optional[str] = field(default=None)
    lca: Optional[dict] = field(default=None)
    
    
    def default():
        return ElectrolyzerConfig(1.0)
    
default_config = ElectrolyzerConfig.default()

@define
class ElectrolyzerPlant(PowerSource):
    site: SiteInfo
    config: ElectrolyzerConfig
    config_name: str = field(init=False, default="DefaultElectrolyzerPlant")
    simple_fin_config: SimpleFinanceConfig = field(default=None)

    def __attrs_post_init__(self):
        """
        ElectrolyzerPlant

        Args:
            site: Site information
            config: Electrolyzer plant configuration
        """
        
        if self.config is None:
            system_model = SimpleElectrolyzer(self.site,default_config)
            financial_model = Singleowner.default('WindPowerSingleOwner')
        else:
            system_model = SimpleElectrolyzer(self.site,self.config)
            if self.config.simple_fin_config:
                financial_model = SimpleFinance(self.config.simple_fin_config)
            else:
                financial_model = Singleowner.default('WindPowerSingleOwner')

        super().__init__("ElectrolyzerPlant", self.site, system_model, financial_model)

        self.capacity_kw = self.config.capacity_kw
        self.system_capacity_kw = self.config.capacity_kw

    @property
    def capacity_kw(self):
        return self._system_model.value("capacity_kw")
    
    @capacity_kw.setter
    def capacity_kw(self, kw: float):
        self._system_model.value("capacity_kw",kw)

    @property
    def system_capacity_kw(self):
        return self._system_model.value("system_capacity_kw")
    
    @system_capacity_kw.setter
    def system_capacity_kw(self, kw: float):
        self._system_model.value("system_capacity_kw",kw)

    @property
    def generation_profile(self):
        return self._system_model.value("generation_profile")
    
    @generation_profile.setter
    def generation_profile(self, kw: float):
        self._system_model.value("generation_profile",kw)

    @property
    def gen(self):
        return self._system_model.value("gen")
    
    @gen.setter
    def gen(self, kw: float):
        self._system_model.value("gen",kw)