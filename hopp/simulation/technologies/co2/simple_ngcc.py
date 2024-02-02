from attrs import define, field
from typing import TYPE_CHECKING
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities.validators import gt_zero
# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.co2.co2_plant import CO2Config

@define
class SimpleNGCC(BaseClass):
    """
    Configuration class for SimpleNGCC.

    Args:
        site = SiteInfo object
        config = CO2Config object
        name = Name
        :param input_streams_kg_s: Dict of flows coming into the plant
        :param output_streams_kg_s: Dict of flows coming out of the plant
        :param input_streams_kw: Dict of power coming into the plant
        :param output_streams_kw: Dict of power coming out of the plant
    """
    site: SiteInfo = field()
    config: "CO2Config" = field()
    name: str = field()
    input_streams_kg_s: dict = {}
    output_streams_kg_s: dict = {}
    input_streams_kw: dict = {}
    output_streams_kw: dict = {}


    def __attrs_post_init__(self):
        self.co2_kg_s = self.config.co2_kg_s
        self.annual_mass_kg = None

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
        Executes a CO2 plant simulation
        '''
        co2_kg_s = self.config.co2_kg_s
        self.output_streams_kg_s["co2"] = co2_kg_s
        self.flow_kg_s = [co2_kg_s]
        self.annual_mass_kg = co2_kg_s*60*60*24*365
        capture_model = self.config.capture_model
        if capture_model == 'AmineScrub':
            kwh_kg = 2.886
        elif capture_model == None:
            kwh_kg = 2.9225
        self.output_streams_kg_s['co2'] = co2_kg_s
        self.output_streams_kw['electricity'] = co2_kg_s*kwh_kg*60*60

@define
class SimpleNGCC_Finance(BaseClass):
    """
    Configuration class for SimpleNGCC_Finance.

    Args:
        config = CO2Config object
        life_yr: lifetime, years
        doll_yr: dollar year that subsequent arguments are reported in
        capex = capital expenses
        fopex_ann = fixed operating expenses per annum
        vopex_kg = variable operating expenses per kg product
        fcr = fixed charged rate
    """
    config: "CO2Config" = field()
    life_yr: int = field(validator=gt_zero, default=30)
    doll_yr: int = field(validator=gt_zero, default=2020)
    capex: float = field(validator=gt_zero, default=1e6)
    fopex_ann: float = field(validator=gt_zero, default=1e3)
    vopex_kg: float = field(validator=gt_zero, default=1e0)
    fcr: float = field(validator=gt_zero, default=0.07)

    test_value = 0.1
    input_dict = {'test':test_value}

    def __attrs_post_init__(self):
        self.cap_kg_s = self.config.co2_kg_s
        self.product = "co2"

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