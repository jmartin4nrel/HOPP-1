from attrs import define, field
from typing import TYPE_CHECKING
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities.validators import gt_zero
# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.grid_sales import GridSalesConfig
    from hopp.simulation.technologies.grid_purchase import GridPurchaseConfig

@define
class SimpleGridSales(BaseClass):
    """
    Configuration class for SimpleGridSales

    Args:
        site = SiteInfo object
        config = GridSalesConfig object
        name = Name
    """
    site: SiteInfo = field()
    config: "GridSalesConfig" = field()

    def __attrs_post_init__(self):
        self.interconnect_kw = self.config.interconnect_kw
        self.generation_profile = self.config.generation_profile
        self.gen = self.generation_profile
        self.system_capacity_kw = self.config.interconnect_kw


    def value(self, name: str, set_value=None):
        """
        if set_value = None, then retrieve value; otherwise overwrite variable's value
        """
        if set_value:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)

    def execute(self, project_life):
        '''
        Executes a grid sales simulation
        '''
        self.annual_energy = sum(self.generation_profile)


@define
class SimpleGridPurchase(BaseClass):
    """
    Configuration class for SimpleGridPurchase

    Args:
        site = SiteInfo object
        config = GridPurcahseConfig object
        name = Name
    """
    site: SiteInfo = field()
    config: "GridPurchaseConfig" = field()

    def __attrs_post_init__(self):
        self.interconnect_kw = self.config.interconnect_kw
        self.generation_profile = self.config.generation_profile
        self.gen = self.generation_profile
        self.system_capacity_kw = self.config.interconnect_kw


    def value(self, name: str, set_value=None):
        """
        if set_value = None, then retrieve value; otherwise overwrite variable's value
        """
        if set_value:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)

    def execute(self, project_life):
        '''
        Executes a grid sales simulation
        '''
        self.annual_energy = sum(self.generation_profile)
