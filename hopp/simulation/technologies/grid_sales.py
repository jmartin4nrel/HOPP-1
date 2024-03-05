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
from hopp.simulation.technologies.simple_grid import SimpleGridSales
from hopp.utilities.log import hybrid_logger as logger


@define
class GridSalesConfig(BaseClass):
    """
    Configuration class for GridSales.

    Args:
        interconnection_kw: The capacity in kw

    """
    interconnect_kw: float = field(default=50000.0, validator=gt_zero)
    model_name: str = field(default="SimpleGridSales", validator=contains(["SimpleGridSales"]))
    generation_profile: list = field(default=[0.0*8760])
    simple_fin_config: Optional[dict] = field(default=None)
    model_input_file: Optional[str] = field(default=None)
    lca: Optional[dict] = field(default=None)
    
    
    def default():
        return GridSalesConfig(1.0)
    
default_config = GridSalesConfig.default()

@define
class GridSales(PowerSource):
    site: SiteInfo
    config: GridSalesConfig
    config_name: str = field(init=False, default="DefaultGridSales")
    simple_fin_config: SimpleFinanceConfig = field(default=None)

    def __attrs_post_init__(self):
        """
        GridSales

        Args:
            site: Site information
            config: Grid Sales configuration
        """
        
        if self.config is None:
            system_model = SimpleGridSales(self.site,default_config)
            financial_model = Singleowner.default('WindPowerSingleOwner')
        else:
            system_model = SimpleGridSales(self.site,self.config)
            if self.config.simple_fin_config:
                financial_model = SimpleFinance(self.config.simple_fin_config)
                financial_model.system_capacity_kw = self.config.interconnect_kw
            else:
                financial_model = Singleowner.default('WindPowerSingleOwner')

        super().__init__("GridSales", self.site, system_model, financial_model)

        self.capacity_kw = self.config.interconnect_kw

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

    @property
    def annual_energy_kwh(self):
        return self._system_model.value("annual_energy")
    
    @annual_energy_kwh.setter
    def annual_energy_kwh(self, kwh: float):
        self._system_model.value("annual_energy",kwh)