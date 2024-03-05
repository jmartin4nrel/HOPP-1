from attrs import define, field
from typing import TYPE_CHECKING
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities.validators import gt_zero
# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.hydrogen.electrolyzer_plant import ElectrolyzerConfig

@define
class SimpleElectrolyzer(BaseClass):
    """
    Configuration class for SimpleElectrolyzer.

    Args:
        site = SiteInfo object
        config = HydrogenConfig object
    """
    site: SiteInfo = field()
    config: "ElectrolyzerConfig" = field()


    def __attrs_post_init__(self):
        self.system_capacity_kw = self.config.system_capacity_kw
        self.generation_profile = self.config.generation_profile
        self.gen = self.generation_profile
        self.input_streams_kg_s = self.config.input_streams_kg_s
        self.output_streams_kg_s = self.config.output_streams_kg_s
        self.kwh_kg_h2 = self.config.kwh_kg_h2
        self.kg_h2o_kg_h2 = self.config.kg_h2o_kg_h2
        self.annual_mass_h2_kg = None
        self.annual_energy = None


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
        Executes an electrolyzer simulation
        '''
        self.output_streams_kg_s['hydrogen'] = [i*self.kwh_kg_h2 for i in self.generation_profile]
        self.input_streams_kg_s['water'] = [i*self.kg_h2o_kg_h2 for i in self.output_streams_kg_s['hydrogen']]
        self.annual_energy = sum(self.generation_profile)


@define
class SimpleElectrolyzerFinance(BaseClass):
    """
    Configuration class for SimpleElectrolyzerFinance.

    Args:
        config = ElectrolyzerConfig object
        life_yr: lifetime, years
        doll_yr: dollar year that subsequent arguments are reported in
        capex = capital expenses
        fopex_ann = fixed operating expenses per annum
        vopex_kwh = variable operating expenses per kwh product
        fcr = fixed charged rate
    """
    config: "ElectrolyzerConfig" = field()
    life_yr: int = field(validator=gt_zero, default=30)
    doll_yr: int = field(validator=gt_zero, default=2020)
    capex: float = field(validator=gt_zero, default=1e6)
    fopex_ann: float = field(validator=gt_zero, default=1e3)
    vopex_kwh: float = field(validator=gt_zero, default=1e0)
    fcr: float = field(validator=gt_zero, default=0.07)

    test_value = 0.1
    input_dict = {'test':test_value}

    def __attrs_post_init__(self):
        self.system_capacity_kw = self.config.system_capacity_kw

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