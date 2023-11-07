from attrs import define, field
from typing import TYPE_CHECKING
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities.validators import gt_zero
# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.fuel.fuel_plant import FuelConfig

@define
class SimpleReactor(BaseClass):
    """
    Configuration class for SimpleReactor.

    Args:
        site = SiteInfo object
        config = FuelConfig object
        name = Name
    """
    site: SiteInfo = field()
    config: "FuelConfig" = field()
    name: str = field()

    def __attrs_post_init__(self):
        self.fuel_prod_kg_s = self.config.fuel_prod_kg_s
        self.fuel_produced = self.config.fuel_produced

    def value(self, name: str, set_value=None):
        """
        if set_value = None, then retrieve value; otherwise overwrite variable's value
        """
        if set_value:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)

@define
class SimpleReactorFinance(BaseClass):
    """
    Configuration class for SimpleReactorFinance.

    Args:
        config = FuelConfig object
        life_yr: lifetime, years
        doll_yr: dollar year that subsequent arguments are reported in
        capex = capital expenses
        fopex_ann = fixed operating expenses per annum
        vopex_kg = variable operating expenses per kg product
        fcr = fixed charged rate
    """
    config: "FuelConfig" = field()
    life_yr: int = field(validator=gt_zero, default=30)
    doll_yr: int = field(validator=gt_zero, default=2020)
    capex: float = field(validator=gt_zero, default=1e6)
    fopex_ann: float = field(validator=gt_zero, default=1e3)
    vopex_kg: float = field(validator=gt_zero, default=1e0)
    fcr: float = field(validator=gt_zero, default=0.07)

    test_value = 0.1
    input_dict = {'test':test_value}

    def __attrs_post_init__(self):
        self.cap_kg_s = self.config.fuel_prod_kg_s
        self.product = self.config.fuel_produced

    def assign(self, input_dict, ignore_missing_vals=False):
        """
        Assign attribues from nested dictionary, except for Outputs

        :param input_dict: nested dictionary of values
        :param ignore_missing_vals: if True, do not throw exception if value not in self
        """
        for k, v in input_dict.items():
            if not isinstance(v, dict):
                try:
                    self.value(k, v)
                except Exception as e:
                    if not ignore_missing_vals:
                        raise IOError(f"{self.__class__}'s attribute {k} could not be set to {v}: {e}")
            elif k == 'Outputs':
                continue    # do not assign from Outputs category
            else:
                self.assign(input_dict[k], ignore_missing_vals)