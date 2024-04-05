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
        :param input_streams_kg_s: Dict of flows coming into the reactor
        :param output_streams_kg_s: Dict of flows coming out of the reactor
    """
    site: SiteInfo = field()
    config: "FuelConfig" = field()
    name: str = field()
    input_streams_kg_s: dict = {}
    output_streams_kg_s: dict = {}
    input_streams_kw: dict = {}
    output_streams_kw: dict = {}
    flow_kg_s: list = [0.0]*8760


    def __attrs_post_init__(self):
        self.fuel_prod_kg_s = self.config.fuel_prod_kg_s
        self.fuel_produced = self.config.fuel_produced
        self.reactor_tech = self.config.reactor_tech
        self.catalyst = self.config.catalyst
        self.annual_mass_kg = None
        self.flow_kg_s = [self.fuel_prod_kg_s]*8760

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
        Executes a fuel plant simulation
        '''
        fuel_kg_s = self.fuel_prod_kg_s
        fuel = self.fuel_produced
        reactor = self.reactor_tech
        catalyst = self.catalyst
        self.output_streams_kg_s[fuel] = [fuel_kg_s]*8760
        self.flow_kg_s = self.output_streams_kg_s[fuel]
        self.annual_mass_kg = fuel_kg_s*60*60*24*365
        kwh_kgH2 = 54.66
        kj_kgH2 = kwh_kgH2*3600
        kgH2O_kgH2 = 14.309
        if fuel == 'methanol':            
            h2ratio = self.config.reaction_params['h2ratio']
            co2ratio = self.config.reaction_params['co2ratio']
            self.input_streams_kg_s['hydrogen'] = [i*h2ratio for i in self.output_streams_kg_s[fuel]]
            self.input_streams_kg_s['water'] = [i*kgH2O_kgH2 for i in self.input_streams_kg_s['hydrogen']]
            self.input_streams_kw['electricity'] = [i*kj_kgH2 for i in self.input_streams_kg_s['hydrogen']]
            self.input_streams_kg_s['carbon dioxide'] = [i*co2ratio for i in self.output_streams_kg_s[fuel]]
        if fuel == 'synthetic natural gas':
            if reactor == 'sabatier':
                h2ratio = 0.125
                co2ratio = 2.75
                kwh_kgH2 = 54.66
                kj_kgH2 = kwh_kgH2*3600
                kgH2O_kgH2 = 14.309
            if reactor == 'RCC recycle':
                if catalyst == 'CZA':
                    h2ratio = 0.33509977
                    co2ratio = 9.1105156
                    kwh_kgH2 = 54.66
                    kj_kgH2 = kwh_kgH2*3600
                    kgH2O_kgH2 = 14.309
            self.input_streams_kg_s['hydrogen'] = [i*h2ratio for i in self.output_streams_kg_s[fuel]]
            self.input_streams_kg_s['water'] = [i*kgH2O_kgH2 for i in self.input_streams_kg_s['hydrogen']]
            self.input_streams_kw['electricity'] = [i*kj_kgH2 for i in self.input_streams_kg_s['hydrogen']]
            self.input_streams_kg_s['carbon dioxide'] = [i*co2ratio for i in self.output_streams_kg_s[fuel]]
            
        

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