from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel
from attrs import define, field
from typing import Optional

@define
class SimpleFinanceConfig(BaseClass):

        cost_tech: str = 'hybrid'
        input_dollar_yr: int = 2020
        output_dollar_yr: int = 2020
        inflation: float = 0.02
        fcr_real: float = 0.07
        tasc_toc: float = 1.1
        toc: float = 0.
        foc_yr: float = 0.
        voc_kg: float = 0.
        voc_kwh: float = 0.
        levelized_cost_kg: float = 10.
        levelized_cost_kwh: float = 1.
        toc_kw: Optional[float] = field(default=None)
        foc_kw_yr: Optional[float] = field(default=None)
        toc_kg_s: Optional[float] = field(default=None)
        foc_kg_s_yr: Optional[float] = field(default=None)
        
        def default():
            return SimpleFinanceConfig(1.0)
        
        def assign(self, input_dict, ignore_missing_vals=False):
            """
            Assign attribues from nested dictionary

            :param input_dict: nested dictionary of values
            :param ignore_missing_vals: if True, do not throw exception if value not in self
            """
            for k, v in input_dict.items():
                if not isinstance(v, dict):
                    try:
                        self.__setattr__(k, v)
                    except Exception as e:
                        if not ignore_missing_vals:
                            raise IOError(f"{self.__class__}'s attribute {k} could not be set to {v}: {e}")
                else:
                    self.assign(input_dict[k], ignore_missing_vals)


default_config = SimpleFinanceConfig.default()

@define
class SimpleFinance(CustomFinancialModel):
    config: SimpleFinanceConfig
    config_name: str = field(init=False, default="DefaultFinanceConfig")
    system_capacity_kw: float = 1e5
    system_capacity_kg_s: float = 1e3

    def __attrs_post_init__(self):

        super().__init__({})

        self.input_dollar_yr = self.config.input_dollar_yr
        self.output_dollar_yr = self.config.output_dollar_yr
        self.inflation = self.config.inflation
        self.fcr_real = self.config.fcr_real
        self.tasc_toc = self.config.tasc_toc
        
        # toc: Total overnight cost (a.k.a. CAPEX)
        self.toc = self.config.toc
        self.toc_kw = self.config.toc_kw
        self.toc_kg_s = self.config.toc_kg_s
        
        # tasc: Total as-spent cost (a.k.a. CAPEX over expenditure period)
        self.tasc = 0.
        
        # foc: Fixed operating cost (a.k.a. fOPEX)
        self.foc_yr = self.config.foc_yr
        self.foc_kw_yr = self.config.foc_kw_yr
        self.foc_kg_s_yr = self.config.foc_kg_s_yr
        
        # voc: Variable operating cost (a.k.a. vOPEX)
        self.voc_kg = self.config.voc_kg
        self.voc_kwh = self.config.voc_kwh
        
        # Levelized cost
        self.lc_kg = 0.
        self.lc_kwh = 0.

    
    def calc_levelized_cost_mass(self, output_kg_yr):
        
        if self.toc_kg_s is not None:
            self.toc = self.toc_kg_s * self.system_capacity_kg_s
        if self.foc_kg_s_yr is not None:
            self.foc_yr = self.foc_kg_s_yr * self.system_capacity_kg_s

        # Correct for inflation
        toc = self.toc*(1+self.inflation)**(self.output_dollar_yr-self.input_dollar_yr)
        self.toc = toc
        foc_yr = self.foc_yr*(1+self.inflation)**(self.output_dollar_yr-self.input_dollar_yr)
        self.foc_yr = foc_yr
        voc_kg = self.voc_kg*(1+self.inflation)**(self.output_dollar_yr-self.input_dollar_yr)
        self.voc_kg = voc_kg
        
        # Apply TASC multiplier (spreading CAPEX over expentidure period)
        tasc = toc*self.tasc_toc
        self.tasc = tasc
        
        # Calculate levelized costs
        self.lc_kg = (tasc*self.fcr_real+foc_yr)/output_kg_yr+voc_kg

        return self.lc_kg

    def calc_levelized_cost_energy(self, output_kwh_yr):
        
        if self.toc_kw is not None:
            self.toc = self.toc_kw * self.system_capacity_kw
        if self.foc_kw_yr is not None:
            self.foc_yr = self.foc_kw_yr * self.system_capacity_kw

        # Correct for inflation
        toc = self.toc*(1+self.inflation)**(self.output_dollar_yr-self.input_dollar_yr)
        self.toc = toc
        foc_yr = self.foc_yr*(1+self.inflation)**(self.output_dollar_yr-self.input_dollar_yr)
        self.foc_yr = foc_yr
        voc_kwh = self.voc_kwh*(1+self.inflation)**(self.output_dollar_yr-self.input_dollar_yr)
        self.voc_kwh = voc_kwh
        
        # Apply TASC multiplier (spreading CAPEX over expentidure period)
        tasc = toc*self.tasc_toc
        self.tasc = tasc
        
        # Calculate levelized costs
        if output_kwh_yr != 0:
            self.lc_kwh = (tasc*self.fcr_real+foc_yr)/output_kwh_yr+voc_kwh
        else:
            self.lc_kwh = 0

        return self.lc_kwh



