from scipy.interpolate import LinearNDInterpolator as interp
from pathlib import Path
import pandas as pd
import numpy as np

from .bos_model import BOSCalculator
from hopp.utilities.log import bos_logger as logger

file_path = Path(__file__).parent


class BOSLookup(BOSCalculator):
    def __init__(self):
        super().__init__()
        self.name = "BOSLookup"

        self.input_parameters = ["Interconnection Capacity",
                                 "Wind Installed Capacity",
                                 "Solar Installed Capacity"]

        # List of desired output parameters from the JSON lookup
        self.desired_output_parameters = ["Wind BOS Cost",
                                          "Solar BOS Cost",
                                          "Total Project Cost"]

        # Loads the json data containing all the BOS cost information from the excel model
        self.data, self.contents = self._load_lookup()
        self.interpolating_fxns = self._load_interp()

        for p in self.desired_output_parameters:
            if p not in self.data.columns:
                raise KeyError(p + " column missing")

    def _load_lookup(self):
        file = file_path / "BOSLookup.csv"
        with open(file, "r") as f:
            data = pd.read_csv(f)
        contents = data[self.input_parameters].values
        return data, contents

    def _load_interp(self):
        fxns = []
        for p in self.desired_output_parameters:
            f = interp(self.contents, self.data[p].values)
            fxns.append(f)
        return fxns

    def _lookup_costs(self, wind_mw, solar_mw, interconnection_mw):
        if wind_mw + solar_mw == 0:
            return 0, 0, 0

        search_inputs = np.array([interconnection_mw, wind_mw, solar_mw])
        distance_norm = np.linalg.norm(self.contents - search_inputs, axis=1)
        min_index = np.argmin(distance_norm)
        min_distance = distance_norm[min_index]

        vals = []
        for i in range(len(self.desired_output_parameters)):
            vals.append(self.interpolating_fxns[i](search_inputs)[0])

        if np.isnan(vals).any():
            wind_bos_cost = self.data.iloc[min_index:min_index+1]["Wind BOS Cost"].values
            solar_bos_cost = self.data.iloc[min_index:min_index+1]["Solar BOS Cost"].values
            if min_distance / np.linalg.norm(search_inputs) > .05:
                Warning("Inputs (Wind Size: {}MW and Solar Size: {}MW) to BOSLookup outside of range and cannot be extrapolated".format(wind_mw, solar_mw))
        else:
            wind_bos_cost = vals[self.desired_output_parameters.index("Wind BOS Cost")]
            solar_bos_cost = vals[self.desired_output_parameters.index("Solar BOS Cost")]

        total_bos_cost = wind_bos_cost + solar_bos_cost
        logger.info("Total BOS Cost: {} Wind BOS Cost: {} Solar BOS Cost {}".
                    format(total_bos_cost, wind_bos_cost, solar_bos_cost))

        return wind_bos_cost, solar_bos_cost, total_bos_cost, min_distance

    def _lookup_project_costs(self, wind_mw, solar_mw, interconnection_mw):
        if wind_mw + solar_mw == 0:
            return 0, 0, 0

        # Lookup sheet does not have interconnection sizes >500 MW
        interconnection_mw = np.min([500,interconnection_mw])

        # When looking up single-tech plant sizes, the interconnect cannot be bigger than the plant
        if solar_mw > 0 and wind_mw == 0:
            interconnection_mw = np.min([solar_mw,interconnection_mw])
        if solar_mw == 0 and wind_mw > 0:
            interconnection_mw = np.min([wind_mw,interconnection_mw])

        search_inputs = np.array([interconnection_mw, wind_mw, solar_mw])
        distance_norm = np.linalg.norm(self.contents - search_inputs, axis=1)
        min_index = np.argmin(distance_norm)
        min_distance = distance_norm[min_index]

        vals = []
        for i in range(len(self.desired_output_parameters)):
            vals.append(self.interpolating_fxns[i](search_inputs)[0])

        if np.isnan(vals).any():
            wind_bos_cost = self.data.iloc[min_index:min_index+1]["Wind BOS Cost"].values
            solar_bos_cost = self.data.iloc[min_index:min_index+1]["Solar BOS Cost"].values
            total_project_cost = self.data.iloc[min_index:min_index+1]["Total Project Cost"].values
            if min_distance / np.linalg.norm(search_inputs) > .05:
                Warning("Inputs (Wind Size: {}MW and Solar Size: {}MW) to BOSLookup outside of range and cannot be extrapolated".format(wind_mw, solar_mw))
        else:
            wind_bos_cost = vals[self.desired_output_parameters.index("Wind BOS Cost")]
            solar_bos_cost = vals[self.desired_output_parameters.index("Solar BOS Cost")]
            total_project_cost = vals[self.desired_output_parameters.index("Total Project Cost")]

        logger.info("Total Project Cost: {} Wind BOS Cost: {} Solar BOS Cost {}".
                    format(total_project_cost, wind_bos_cost, solar_bos_cost))

        return wind_bos_cost, solar_bos_cost, total_project_cost, min_distance
    
    def calculate_bos_costs(self, wind_mw, solar_mw, interconnection_mw, scenario='greenfield'):
        """
        Calls the appropriate calculate_bos_costs_x method for the Cost Source data specified

        :param wind_mw: Installed Capacity (MW) of wind component
        :param solar_mw: Installed Capacity (MW) of solar component
        :param interconnection_mw:
        :param scenario: 'greenfield' or 'solar addition'
        :return: wind, solar and total bos cost
        """
        scenario = scenario.lower()
        if scenario == 'greenfield':
            return self._lookup_costs(wind_mw, solar_mw, interconnection_mw)
        elif scenario == 'simple financial':
            return self._lookup_project_costs(wind_mw, solar_mw, interconnection_mw)
        elif scenario == 'solar addition':
            raise NotImplementedError
        else:
            raise ValueError("scenario type {} not recognized".format(scenario))
