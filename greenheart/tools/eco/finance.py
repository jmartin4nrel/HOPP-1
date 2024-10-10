import os.path

import numpy as np
import numpy_financial as npf

import ProFAST  # system financial model
from ORBIT import ProjectManager
import pandas as pd

from typing import Dict, Union, Optional

from attrs import define, Factory, field

from hopp.simulation import HoppInterface
import greenheart.tools.profast_tools as pf_tools
import greenheart.tools.eco.cost_tools as cost_tools

from greenheart.simulation.technologies.hydrogen.electrolysis.custom_electrolysis_costs import (
    summarize_electrolysis_cost_and_performance
)
from hopp.utilities.utilities import write_yaml

@define
class WindCostConfig:
    """
    Represents the inputs to the wind cost models

    Attributes:
        design_scenario (Dict[str, str]):
            Definition of plant subsystem locations (e.g. onshore platform, offshore, none, etc)
        hopp_config (Dict[str, float]):
            Configuration parameters for HOPP
        greenheart_config (Dict[str, float]):
            Configuration parameters for Greenheart
        orbit_config (Dict[str, float], optional):
            Required input structure for ORBIT
        turbine_config (Dict[str, float], optional):
            Configuration parameters specific to turbine
        orbit_hybrid_electrical_export_config (Dict[str, float], optional):
            Configuration parameters for hybrid electrical export in ORBIT, required if using a different substation size for the hybrid plant than for the wind plant alone
        weather (Union[list, tuple, numpy.ndarray], optional):
            Array-like of wind speeds for ORBIT to use in determining installation time and costs
    """

    design_scenario: Dict[str, str]
    hopp_config: Dict[str, float]
    greenheart_config: Dict[str, float]
    orbit_config: Optional[Dict[str, float]] = field(default={})
    turbine_config: Optional[Dict[str, float]] = field(default={})
    orbit_hybrid_electrical_export_config: Optional[Dict[str, float]] = field(
        default={}
    )
    weather: Optional[Union[list, tuple, np.ndarray]] = field(default=None)
    hopp_interface: Optional[HoppInterface] = field(default=None)


@define
class WindCostOutputs:
    """
    Represents the outputs to the wind cost models.

    Attributes:
        total_wind_cost_no_export (float):
            Total wind cost without export system costs
        total_used_export_system_costs (float):
            Total used export system costs
        annual_operating_cost_wind (float):
            Annual operating cost for wind
        installation_time (float, optional):
            Estimated installation time in months (default: 0.0)
        orbit_project (dict, optional):
            Details of the ORBIT project (default: None)
    """

    total_wind_cost_no_export: float
    annual_operating_cost_wind: float
    installation_time: float = field(default=0.0)
    total_used_export_system_costs: Optional[float] = field(default=0.0)
    orbit_project: Optional[Union[dict, ProjectManager]] = field(default=None)


def run_wind_cost_model(
    wind_cost_inputs: WindCostConfig, verbose=False
) -> WindCostOutputs:
    if "wind" in wind_cost_inputs.hopp_config["technologies"]:
        if wind_cost_inputs.design_scenario["wind_location"] == "offshore":

            # if per kw
            project, orbit_hybrid_electrical_export_project = run_orbit(
                wind_cost_inputs.orbit_config,
                verbose=verbose,
                weather=wind_cost_inputs.weather,
                orbit_hybrid_electrical_export_config=wind_cost_inputs.orbit_hybrid_electrical_export_config,
            )

            (
                total_wind_cost_no_export,
                total_used_export_system_costs,
            ) = breakout_export_costs_from_orbit_results(
                project,
                wind_cost_inputs.greenheart_config,
                wind_cost_inputs.design_scenario,
            )

            if orbit_hybrid_electrical_export_project is not None:
                (
                    _,
                    total_used_export_system_costs,
                ) = breakout_export_costs_from_orbit_results(
                    orbit_hybrid_electrical_export_project,
                    wind_cost_inputs.greenheart_config,
                    wind_cost_inputs.design_scenario,
                )

            # WIND ONLY Total O&M expenses including fixed, variable, and capacity-based, $/year
            # use values from hybrid substation if a hybrid plant
            if orbit_hybrid_electrical_export_project is None:

                annual_operating_cost_wind = (
                    max(project.monthly_opex.values()) * 12
                )  # np.average(hopp_results["hybrid_plant"].wind.om_total_expense)

            else:

                annual_operating_cost_wind = (
                    max(orbit_hybrid_electrical_export_project.monthly_opex.values()) * 12
                )

            if (
                "installation_time"
                in wind_cost_inputs.greenheart_config["project_parameters"]
            ):
                installation_time = wind_cost_inputs.greenheart_config[
                    "project_parameters"
                ]["installation_time"]
            else:
                installation_time = (project.installation_time / (365 * 24)) * (12.0 / 1.0)

            # if total amount
            # TODO
            return WindCostOutputs(
                total_wind_cost_no_export=total_wind_cost_no_export,
                total_used_export_system_costs=total_used_export_system_costs,
                annual_operating_cost_wind=annual_operating_cost_wind,
                installation_time=installation_time,
                orbit_project=project,
            )
        elif wind_cost_inputs.design_scenario["wind_location"] == "onshore":
            total_wind_cost_no_export = (
                wind_cost_inputs.hopp_config["config"]["cost_info"][
                    "wind_installed_cost_mw"
                ]
                * wind_cost_inputs.hopp_config["technologies"]["wind"]["num_turbines"]
                * wind_cost_inputs.turbine_config["turbine_rating"]
            )

            annual_operating_cost_wind = wind_cost_inputs.hopp_interface.system.wind.om_total_expense[
                0
            ]

            if (
                "installation_time"
                in wind_cost_inputs.greenheart_config["project_parameters"]
            ):
                installation_time = wind_cost_inputs.greenheart_config[
                    "project_parameters"
                ]["installation_time"]
            else:
                installation_time = 0

            return WindCostOutputs(
                total_wind_cost_no_export=total_wind_cost_no_export,
                annual_operating_cost_wind=annual_operating_cost_wind,
                installation_time=installation_time,
            )
        else:
            raise ValueError(
                "Wind design location must either be 'onshore' or 'offshore', but currently "
                f"'wind_location' is set to {wind_cost_inputs.design_scenario['wind_location']}."
            )
    else:
        return WindCostOutputs(
                total_wind_cost_no_export=0,
                annual_operating_cost_wind=0,
                installation_time=36,
            )



# Function to run orbit from provided inputs - this is just for wind costs
def run_orbit(
    orbit_config, verbose=False, weather=None, orbit_hybrid_electrical_export_config={}
):
    # set up ORBIT
    project = ProjectManager(orbit_config, weather=weather)

    # run ORBIT
    project.run(availability=orbit_config["installation_availability"])

    # run ORBIT for hybrid substation if applicable
    if orbit_hybrid_electrical_export_config == {}:
        hybrid_substation_project = None
    else:
        hybrid_substation_project = ProjectManager(
            orbit_hybrid_electrical_export_config, weather=weather
        )
        hybrid_substation_project.run(
            availability=orbit_config["installation_availability"]
        )

    # print results if desired
    if verbose:
        print(f"Installation CapEx:  {project.installation_capex/1e6:.0f} M")
        print(f"System CapEx:        {project.system_capex/1e6:.0f} M")
        print(f"Turbine CapEx:       {project.turbine_capex/1e6:.0f} M")
        print(f"Soft CapEx:          {project.soft_capex/1e6:.0f} M")
        print(f"Total CapEx:        {project.total_capex/1e6:.0f} M")
        print(f"Annual OpEx Rate:        {max(project.monthly_opex.values())*12:.0f} ")
        print(f"\nInstallation Time: {project.installation_time:.0f} h")
        print("\nN Substations: ", (project.phases["ElectricalDesign"].num_substations))
        print("N cables: ", (project.phases["ElectricalDesign"].num_cables))
        print("\n")

        # cable cost breakdown
        print("Cable specific costs")
        print(
            "Export cable installation CAPEX: %.2f M USD"
            % (project.phases["ExportCableInstallation"].installation_capex * 1e-6)
        )
        print("\n")

    return project, hybrid_substation_project


def adjust_orbit_costs(orbit_project, greenheart_config):

    if ("expected_plant_cost" in greenheart_config["finance_parameters"]["wind"]) and (
        greenheart_config["finance_parameters"]["wind"]["expected_plant_cost"] != "none"
    ):
        wind_capex_multiplier = (
            greenheart_config["finance_parameters"]["wind"]["expected_plant_cost"] * 1e9
        ) / orbit_project.total_capex
    else:
        wind_capex_multiplier = 1.0

    wind_total_capex = orbit_project.total_capex * wind_capex_multiplier
    wind_capex_breakdown = orbit_project.capex_breakdown
    for key in wind_capex_breakdown.keys():
        wind_capex_breakdown[key] *= wind_capex_multiplier

    return wind_total_capex, wind_capex_breakdown, wind_capex_multiplier


def breakout_export_costs_from_orbit_results(
    orbit_project, greenheart_config, design_scenario
):
    # adjust wind capex to meet expectations
    wind_total_capex, wind_capex_breakdown, wind_capex_multiplier = adjust_orbit_costs(
        orbit_project=orbit_project, greenheart_config=greenheart_config
    )

    # onshore substation cost is not included in ORBIT costs by default, so we have to add it separately
    total_wind_installed_costs_with_export = wind_total_capex

    # breakout export system costs
    array_cable_equipment_cost = wind_capex_breakdown["Array System"]
    array_cable_installation_cost = wind_capex_breakdown["Array System Installation"]
    total_array_cable_system_capex = (
        array_cable_equipment_cost + array_cable_installation_cost
    )

    export_cable_equipment_cost = wind_capex_breakdown[
        "Export System"
    ]  # this should include the onshore substation
    export_cable_installation_cost = wind_capex_breakdown["Export System Installation"]
    substation_equipment_cost = wind_capex_breakdown["Offshore Substation"]
    substation_installation_cost = wind_capex_breakdown[
        "Offshore Substation Installation"
    ]
    total_export_cable_system_capex = (
        export_cable_equipment_cost + export_cable_installation_cost
    )

    total_offshore_substation_capex = (
        substation_equipment_cost + substation_installation_cost
    )

    total_electrical_export_system_cost = (
        total_array_cable_system_capex
        + total_offshore_substation_capex
        + total_export_cable_system_capex
    )

    ## adjust wind cost to remove export
    if design_scenario["transportation"] == "hvdc+pipeline":
        unused_export_system_cost = 0.0
    elif (
        design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "turbine"
    ):
        unused_export_system_cost = (
            total_array_cable_system_capex
            + total_export_cable_system_capex
            + total_offshore_substation_capex
        )
    elif (
        design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "platform"
    ):
        unused_export_system_cost = (
            total_export_cable_system_capex  # TODO check assumptions here
        )
    elif (
        design_scenario["electrolyzer_location"] == "platform"
        and design_scenario["h2_storage_location"] == "platform"
    ):
        unused_export_system_cost = (
            total_export_cable_system_capex  # TODO check assumptions here
        )
    elif (
        design_scenario["electrolyzer_location"] == "platform"
        or design_scenario["electrolyzer_location"] == "turbine"
    ) and design_scenario["h2_storage_location"] == "onshore":
        unused_export_system_cost = (
            total_export_cable_system_capex  # TODO check assumptions here
        )
    else:
        unused_export_system_cost = 0.0

    total_used_export_system_costs = (
        total_electrical_export_system_cost - unused_export_system_cost
    )

    total_wind_cost_no_export = (
        total_wind_installed_costs_with_export - total_used_export_system_costs
    )

    return total_wind_cost_no_export, total_used_export_system_costs


def run_capex(
    hopp_results,
    wind_cost_results,
    electrolyzer_cost_results,
    h2_pipe_array_results,
    h2_transport_compressor_results,
    h2_transport_pipe_results,
    h2_storage_results,
    hopp_config,
    greenheart_config,
    design_scenario,
    desal_results,
    platform_results,
    verbose=False,
):

    # total_wind_cost_no_export, total_used_export_system_costs = breakout_export_costs_from_orbit_results(orbit_project, greenheart_config, design_scenario)

    # if orbit_hybrid_electrical_export_project is not None:
    #     _, total_used_export_system_costs = breakout_export_costs_from_orbit_results(orbit_hybrid_electrical_export_project, greenheart_config, design_scenario)

    # wave capex
    if hopp_config["site"]["wave"]:
        cost_dict = hopp_results["hybrid_plant"].wave.mhk_costs.cost_outputs

        wcapex = (
            cost_dict["structural_assembly_cost_modeled"]
            + cost_dict["power_takeoff_system_cost_modeled"]
            + cost_dict["mooring_found_substruc_cost_modeled"]
        )
        wbos = (
            cost_dict["development_cost_modeled"]
            + cost_dict["eng_and_mgmt_cost_modeled"]
            + cost_dict["plant_commissioning_cost_modeled"]
            + cost_dict["site_access_port_staging_cost_modeled"]
            + cost_dict["assembly_and_install_cost_modeled"]
            + cost_dict["other_infrastructure_cost_modeled"]
        )
        welec_infrastruc_costs = (
            cost_dict["array_cable_system_cost_modeled"]
            + cost_dict["export_cable_system_cost_modeled"]
            + cost_dict["other_elec_infra_cost_modeled"]
        )  # +\
        # cost_dict['onshore_substation_cost_modeled']+\
        # cost_dict['offshore_substation_cost_modeled']
        # financial = cost_dict['project_contingency']+\
        # cost_dict['insurance_during_construction']+\
        # cost_dict['reserve_accounts']
        wave_capex = wcapex + wbos + welec_infrastruc_costs
    else:
        wave_capex = 0.0

    #wind capex
    if "wind" in hopp_config["technologies"].keys():
        wind_capex = hopp_results["hybrid_plant"].wind.total_installed_cost
    
        if wind_cost_results is not None:
            wind_capex = wind_cost_results.total_wind_cost_no_export
            wind_export_cost = wind_cost_results.total_used_export_system_costs
        else:
            wind_export_cost = 0.0
    else:
            wind_capex = 0.0
            wind_export_cost = 0.0
    # solar capex
    if "pv" in hopp_config["technologies"].keys():
        solar_capex = hopp_results["hybrid_plant"].pv.total_installed_cost
    else:
        solar_capex = 0.0

    # battery capex
    if "battery" in hopp_config["technologies"].keys():
        battery_capex = hopp_results["hybrid_plant"].battery.total_installed_cost
    else:
        battery_capex = 0.0

    # TODO bos capex
    # bos_capex = hopp_results["hybrid_plant"].bos.total_installed_cost

    ## desal capex
    if desal_results != None:
        desal_capex = desal_results["desal_capex_usd"]
    else:
        desal_capex = 0.0

    ## electrolyzer capex
    electrolyzer_total_capital_cost = electrolyzer_cost_results[
        "electrolyzer_total_capital_cost"
    ]

    if (
        design_scenario["electrolyzer_location"] == "platform"
        or design_scenario["h2_storage_location"] == "platform"
        or hopp_config["site"]["solar"]
    ):
        platform_costs = platform_results["capex"]
    else:
        platform_costs = 0.0

    # h2 transport
    h2_transport_compressor_capex = h2_transport_compressor_results["compressor_capex"]
    h2_transport_pipe_capex = h2_transport_pipe_results["total capital cost [$]"][0]

    ## h2 storage
    if greenheart_config["h2_storage"]["type"] == "none":
        h2_storage_capex = 0.0
    elif (
        greenheart_config["h2_storage"]["type"] == "pipe"
    ):  # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "turbine"
    ):  # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "pressure_vessel"
    ):  # pressure vessel storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "salt_cavern"
    ):  # salt cavern storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "lined_rock_cavern"
    ):  # lined rock cavern storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    else:
        raise NotImplementedError(
            "the storage type you have indicated (%s) has not been implemented."
            % greenheart_config["h2_storage"]["type"]
        )

    # store capex component breakdown
    capex_breakdown = {
        # "wind": wind_cost_results.total_wind_cost_no_export,
        "wind": wind_capex,
        "wave": wave_capex,
        "solar": solar_capex,
        "battery": battery_capex,
        "platform": platform_costs,
        # "electrical_export_system": wind_cost_results.total_used_export_system_costs,
        "electrical_export_system": wind_export_cost,
        "desal": desal_capex,
        "electrolyzer": electrolyzer_total_capital_cost,
        "h2_pipe_array": h2_pipe_array_results["capex"],
        "h2_transport_compressor": h2_transport_compressor_capex,
        "h2_transport_pipeline": h2_transport_pipe_capex,
        "h2_storage": h2_storage_capex,
    }

    # discount capex to appropriate year for unified costing
    for key in capex_breakdown.keys():
        if key == "h2_storage":
            # if design_scenario["h2_storage_location"] == "turbine" and greenheart_config["h2_storage"]["type"] == "turbine":
            #     cost_year = greenheart_config["finance_parameters"]["discount_years"][key][
            #         design_scenario["h2_storage_location"]
            #     ]
            # else:
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key][
                greenheart_config["h2_storage"]["type"]
            ]
        else:
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key]

        periods = greenheart_config["project_parameters"]["cost_year"] - cost_year

        capex_breakdown[key] = -npf.fv(
            greenheart_config["finance_parameters"]["costing_general_inflation"],
            periods,
            0.0,
            capex_breakdown[key],
        )

    total_system_installed_cost = sum(
        capex_breakdown[key] for key in capex_breakdown.keys()
    )

    if verbose:
        print("\nCAPEX Breakdown")
        for key in capex_breakdown.keys():
            print(key, "%.2f" % (capex_breakdown[key] * 1e-6), " M")

        print(
            "\nTotal system CAPEX: ",
            "$%.2f" % (total_system_installed_cost * 1e-9),
            " B",
        )

    return total_system_installed_cost, capex_breakdown


def run_opex(
    hopp_results,
    wind_cost_results,
    electrolyzer_cost_results,
    h2_pipe_array_results,
    h2_transport_compressor_results,
    h2_transport_pipe_results,
    h2_storage_results,
    hopp_config,
    greenheart_config,
    desal_results,
    platform_results,
    verbose=False,
    total_export_system_cost=0,
):
    # WIND ONLY Total O&M expenses including fixed, variable, and capacity-based, $/year
    # use values from hybrid substation if a hybrid plant
    # if orbit_hybrid_electrical_export_project is None:

    # wave opex
    if hopp_config["site"]["wave"]:
        cost_dict = hopp_results["hybrid_plant"].wave.mhk_costs.cost_outputs
        wave_opex = cost_dict["maintenance_cost"] + cost_dict["operations_cost"]
    else:
        wave_opex = 0.0

    # solar opex
    if "pv" in hopp_config["technologies"].keys():
        solar_opex = hopp_results["hybrid_plant"].pv.om_total_expense[0]
        if solar_opex < 0.1:
            raise (RuntimeWarning(f"Solar OPEX returned as {solar_opex}"))
    else:
        solar_opex = 0.0

    if wind_cost_results is None:
        if "wind" in hopp_config["technologies"].keys():
            wind_opex = hopp_results["hybrid_plant"].wind.om_total_expense[0]
            if wind_opex < 0.1:
                raise (RuntimeWarning(f"Wind OPEX returned as {wind_opex}"))
        else:
            wind_opex = 0.0
    else:
        wind_opex = wind_cost_results.annual_operating_cost_wind


    # battery opex
    if "battery" in hopp_config["technologies"].keys():
        battery_opex = hopp_results["hybrid_plant"].battery.om_total_expense[0]
        if battery_opex < 0.1:
            raise (RuntimeWarning(f"Battery OPEX returned as {battery_opex}"))
    else:
        battery_opex = 0.0

    # H2 OPEX
    platform_operating_costs = platform_results["opex"]  # TODO update this

    annual_operating_cost_h2 = electrolyzer_cost_results["electrolyzer_OM_cost_annual"]

    h2_transport_compressor_opex = h2_transport_compressor_results[
        "compressor_opex"
    ]  # annual

    h2_transport_pipeline_opex = h2_transport_pipe_results["annual operating cost [$]"][
        0
    ]  # annual

    storage_opex = h2_storage_results["storage_opex"]
    # desal OPEX
    if desal_results != None:
        desal_opex = desal_results["desal_opex_usd_per_year"]
    else:
        desal_opex = 0.0
    annual_operating_cost_desal = desal_opex

    # store opex component breakdown
    opex_breakdown_annual = {
        # "wind_and_electrical": wind_cost_results.annual_operating_cost_wind,
        "wind_and_electrical": wind_opex,
        "platform": platform_operating_costs,
        #   "electrical_export_system": total_export_om_cost,
        "wave": wave_opex,
        "solar": solar_opex,
        "battery": battery_opex,
        "desal": annual_operating_cost_desal,
        "electrolyzer": annual_operating_cost_h2,
        "h2_pipe_array": h2_pipe_array_results["opex"],
        "h2_transport_compressor": h2_transport_compressor_opex,
        "h2_transport_pipeline": h2_transport_pipeline_opex,
        "h2_storage": storage_opex,
    }

    # discount opex to appropriate year for unified costing
    for key in opex_breakdown_annual.keys():
        if key == "h2_storage":
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key][
                greenheart_config["h2_storage"]["type"]
            ]
        else:
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key]

        periods = greenheart_config["project_parameters"]["cost_year"] - cost_year
        opex_breakdown_annual[key] = -npf.fv(
            greenheart_config["finance_parameters"]["costing_general_inflation"],
            periods,
            0.0,
            opex_breakdown_annual[key],
        )

    # Calculate the total annual OPEX of the installed system
    total_annual_operating_costs = sum(opex_breakdown_annual.values())

    if verbose:
        print("\nAnnual OPEX Breakdown")
        for key in opex_breakdown_annual.keys():
            print(key, "%.2f" % (opex_breakdown_annual[key] * 1e-6), " M")

        print(
            "\nTotal Annual OPEX: ",
            "$%.2f" % (total_annual_operating_costs * 1e-6),
            " M",
        )
        print(opex_breakdown_annual)
    return total_annual_operating_costs, opex_breakdown_annual


def run_profast_lcoe(
    greenheart_config,
    wind_cost_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    incentive_option,
    design_scenario,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
):
    gen_inflation = greenheart_config["finance_parameters"]["profast_general_inflation"]

    if "profast_config" in greenheart_config["finance_parameters"]:
        pf_config = greenheart_config["finance_parameters"]["profast_config"]
        analysis_start_year = pf_config["params"]["analysis start year"]
        installation_period_months = pf_config["params"]["installation months"]
        # pf_desc = "Using_PFConfig"
    else:
        # pf_desc = "Using_financeparams"
        pf_config = {"params":{}}
        analysis_start_year = greenheart_config["project_parameters"]["atb_year"] + 1
        installation_period_months = wind_cost_results.installation_time
        pf_config["params"].update({"analysis start year":analysis_start_year})
        pf_config["params"].update({"installation months": installation_period_months})
        pf_config["params"].update({"sales tax":greenheart_config["finance_parameters"]["sales_tax_rate"]})
        pf_config["params"].update({"property tax and insurance":greenheart_config["finance_parameters"]["property_tax"] + 
        greenheart_config["finance_parameters"]["property_insurance"]})
        pf_config["params"].update({"admin expense":greenheart_config["finance_parameters"]["administrative_expense_percent_of_sales"]})
        pf_config["params"].update({"total income tax rate": greenheart_config["finance_parameters"]["total_income_tax_rate"]})
        pf_config["params"].update({"capital gains tax rate":greenheart_config["finance_parameters"]["capital_gains_tax_rate"]})
        pf_config["params"].update({"debt equity ratio of initial financing":greenheart_config["finance_parameters"]["debt_equity_ratio"]})
        pf_config["params"].update({"debt type": greenheart_config["finance_parameters"]["debt_type"]})
        
        pf_config["params"].update({"loan period if used": greenheart_config["finance_parameters"]["loan_period"]})
        pf_config["params"].update({"debt interest rate":greenheart_config["finance_parameters"]["debt_interest_rate"]})
        pf_config["params"].update({"cash onhand":greenheart_config["finance_parameters"]["cash_onhand_months"]})
        pf_config["params"].update({"leverage after tax nominal discount rate":greenheart_config["finance_parameters"]["discount_rate"]})

    pf_config["params"].update({"general inflation rate": gen_inflation})
    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        if 'land_cost' in greenheart_config['finance_parameters']:
            land_cost = greenheart_config['finance_parameters']['land_cost']
        else:
            land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf_config["params"]["commodity"] = {"name": "electricity","unit": "kWh","initial price": 100,"escalation":gen_inflation}
    pf_config["params"]["capacity"] = np.sum(hopp_results["combined_hybrid_power_production_hopp"]) / 365.0
    pf_config["params"]["operating life"] = greenheart_config["project_parameters"]["project_lifetime"]
    pf_config["params"]["long term utilization"] = 1  # TODO should use utilization
    
    pf_config["params"]["non depr assets"] = land_cost
    pf_config["params"]["end of proj sale non depr assets"] = land_cost * (1 + gen_inflation)** greenheart_config["project_parameters"]["project_lifetime"]

    pf = ProFAST.ProFAST()
    params = pf_config['params']
    for i in params:
        pf.set_params(i,params[i])
    
    lcoe_components = ["battery","solar","wind","wave"]

    if design_scenario["transportation"] == "hvdc+pipeline" or not (
        design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "turbine"
    ):
        lcoe_components += ["electrical_export_system"]
    # -------------------------------------- Add capital costs--------------------------------
    depr_type=greenheart_config["finance_parameters"]["depreciation_method"]
    depr_period=greenheart_config["finance_parameters"]["depreciation_period"]
    capital_items = {}
    for item in lcoe_components:
        if any(i in item for i in lcoe_components):
            if capex_breakdown[item]>0:

                capital_item = cost_tools.make_profast_capital_item(capex_breakdown[item],item,refurb=[0])
                capital_items.update(capital_item)
    capital_items = pf_tools.update_defaults(capital_items,'depr_type',depr_type)
    capital_items = pf_tools.update_defaults(capital_items,"depr_period",depr_period)
    
    pf_config["capital_items"] = capital_items
    variables = pf_config['capital_items']
    for i in variables:
        pf.add_capital_item(i,variables[i]["cost"],variables[i]["depr_type"],variables[i]["depr_period"],variables[i]["refurb"])

    # -------------------------------------- Add fixed costs--------------------------------
    fixed_items = {}
    for item in opex_breakdown.keys():
        if any(i in item for i in lcoe_components):
            if opex_breakdown[item]>0:
                fixed_item = cost_tools.make_profast_fixed_cost_item(opex_breakdown[item],item)
                fixed_items.update(fixed_item)
    fixed_items = pf_tools.update_defaults(fixed_items,"escalation",gen_inflation)
    pf_config["fixed_items"] = fixed_items

    variables = pf_config['fixed_items']
    for i in variables:
        pf.add_fixed_cost(i,variables[i]["usage"],variables[i]["unit"],variables[i]["cost"],variables[i]["escalation"])
    
    # ------------------------------------- add incentives -----------------------------------
    """ Note: ptc units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # catch incentive option and add relevant incentives
    incentive_dict = greenheart_config["policy_parameters"][
        "option%s" % (incentive_option)
    ]
    # add electricity_ptc ($/kW)
    # adjust from 1992 dollars to start year
    wind_ptc_in_dollars_per_kw = -npf.fv(
        greenheart_config['finance_parameters']['costing_general_inflation'],
        greenheart_config["project_parameters"]["atb_year"]
        + round((installation_period_months / 12))
        - greenheart_config["finance_parameters"]["discount_years"]["electricity_ptc"],
        0,
        incentive_dict["electricity_ptc"],
    )  # given in 1992 dollars but adjust for inflation

    pf.add_incentive(
        name="Electricity PTC",
        value=wind_ptc_in_dollars_per_kw,
        decay=-gen_inflation,
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

    sol = pf.solve_price()

    lcoe = sol["price"]

    if verbose:
        print("\nProFAST LCOE: ", "%.2f" % (lcoe * 1e3), "$/MWh")

    if show_plots or save_plots:
        savepath = output_dir + "figures/wind_only/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        pf.plot_costs_yearly(
            per_kg=False,
            scale="M",
            remove_zeros=True,
            remove_depreciation=False,
            fileout=savepath
            + "annual_cash_flow_wind_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_costs_yearly2(
            per_kg=False,
            scale="M",
            remove_zeros=True,
            remove_depreciation=False,
            fileout=savepath
            + "annual_cash_flow_wind_only_%i.html" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_capital_expenses(
            fileout=savepath + "capital_expense_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout=savepath + "cash_flow_wind_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_costs(
            fileout=savepath + "cost_breakdown_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )

    return lcoe, pf


def run_profast_grid_only(
    greenheart_config,
    wind_cost_results,
    electrolyzer_physics_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    design_scenario,
    total_accessory_power_renewable_kw,
    total_accessory_power_grid_kw,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
):
    
    electrolyzer_cost_info = summarize_electrolysis_cost_and_performance(electrolyzer_physics_results,greenheart_config["electrolyzer"])
    gen_inflation = greenheart_config["finance_parameters"]["profast_general_inflation"]

    if "feedstock_region" in greenheart_config["site"]:
        feedstock_region = greenheart_config["site"]["feedstock_region"]
    else:
        feedstock_region = "US Average"
    if "profast_config" in greenheart_config["finance_parameters"]:
        pf_config = greenheart_config["finance_parameters"]["profast_config"]
        analysis_start_year = pf_config["params"]["analysis start year"]
        installation_period_months = pf_config["params"]["installation months"]
    else:
        pf_config = {"params":{}}
        analysis_start_year = greenheart_config["project_parameters"]["atb_year"] + 1
        installation_period_months = wind_cost_results.installation_time
        pf_config["params"].update({"analysis start year":analysis_start_year})
        pf_config["params"].update({"installation months": installation_period_months})
        pf_config["params"].update({"sales tax":greenheart_config["finance_parameters"]["sales_tax_rate"]})
        pf_config["params"].update({"property tax and insurance":greenheart_config["finance_parameters"]["property_tax"] + 
        greenheart_config["finance_parameters"]["property_insurance"]})
        pf_config["params"].update({"admin expense":greenheart_config["finance_parameters"]["administrative_expense_percent_of_sales"]})
        pf_config["params"].update({"total income tax rate": greenheart_config["finance_parameters"]["total_income_tax_rate"]})
        pf_config["params"].update({"capital gains tax rate":greenheart_config["finance_parameters"]["capital_gains_tax_rate"]})
        pf_config["params"].update({"debt equity ratio of initial financing":greenheart_config["finance_parameters"]["debt_equity_ratio"]})
        pf_config["params"].update({"debt type": greenheart_config["finance_parameters"]["debt_type"]})
        
        pf_config["params"].update({"loan period if used": greenheart_config["finance_parameters"]["loan_period"]})
        pf_config["params"].update({"debt interest rate":greenheart_config["finance_parameters"]["debt_interest_rate"]})
        pf_config["params"].update({"cash onhand":greenheart_config["finance_parameters"]["cash_onhand_months"]})
        pf_config["params"].update({"leverage after tax nominal discount rate":greenheart_config["finance_parameters"]["discount_rate"]})

    years_of_operation = cost_tools.create_years_of_operation(greenheart_config["project_parameters"]["project_lifetime"],analysis_start_year,installation_period_months)
    electrolyzer_cost_info = summarize_electrolysis_cost_and_performance(electrolyzer_physics_results,greenheart_config["electrolyzer"])
    pf_config["params"].update({"general inflation rate": gen_inflation})
    

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        if 'land_cost' in greenheart_config['finance_parameters']:
            land_cost = greenheart_config['finance_parameters']['land_cost']
        else:
            land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf_config["params"]["commodity"] = {"name": "Hydrogen","unit": "kg","initial price": 100,"escalation":gen_inflation}
    pf_config["params"]["capacity"] = electrolyzer_cost_info["electrolyzer_capacity_kg_pr_day"]
    pf_config["params"]["operating life"] = greenheart_config["project_parameters"]["project_lifetime"]
    utilization = dict(zip(years_of_operation,electrolyzer_cost_info["electrolyzer_utilization"]))
    pf_config["params"]["long term utilization"] = utilization  # TODO should use utilization
    
    pf_config["params"]["non depr assets"] = land_cost
    pf_config["params"]["end of proj sale non depr assets"] = land_cost * (1 + gen_inflation)** greenheart_config["project_parameters"]["project_lifetime"]
    depr_type = greenheart_config["finance_parameters"]["depreciation_method"]
    depr_period = greenheart_config["finance_parameters"]["depreciation_period"]

    pf = ProFAST.ProFAST()
    params = pf_config['params']
    for i in params:
        pf.set_params(i,params[i])

    grid_only_items = ["electrolyzer","h2_storage"]
    # -------------------------------------- Add capital costs--------------------------------
    capital_items = {}
    for item in grid_only_items:
        if capex_breakdown[item]>0:
            if item == "electrolyzer":
                refurb = electrolyzer_cost_info["refurb_cost_simple"]
            else:
                refurb = [0]
            capital_item = cost_tools.make_profast_capital_item(capex_breakdown[item],item,refurb=refurb)
            capital_items.update(capital_item)
    capital_items = pf_tools.update_defaults(capital_items,'depr_type',depr_type)
    capital_items = pf_tools.update_defaults(capital_items,"depr_period",depr_period)
    
    pf_config["capital_items"] = capital_items

    variables = pf_config['capital_items']
    for i in variables:
        pf.add_capital_item(i,variables[i]["cost"],variables[i]["depr_type"],variables[i]["depr_period"],variables[i]["refurb"])
    
    # pf.add_capital_item(name ="Desalination system",cost=capex_breakdown["desal"], depr_type=greenheart_config["finance_parameters"]["depreciation_method"],depr_period=greenheart_config["finance_parameters"]["depreciation_period"],refurb=[0])

    # -------------------------------------- Add fixed costs--------------------------------
    # pf.add_fixed_cost(name="Wind Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["wind"],escalation=gen_inflation)
    # pf.add_fixed_cost(name="Electrical Export Fixed O&M Cost", usage=1.0,unit='$/year',cost=opex_breakdown["electrical_export_system"],escalation=gen_inflation)
    # pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["desal"],escalation=gen_inflation)
    fixed_items = {}
    for item in grid_only_items:
        if opex_breakdown[item]>0:
            fixed_item = cost_tools.make_profast_fixed_cost_item(opex_breakdown[item],item)
            fixed_items.update(fixed_item)
    fixed_items = pf_tools.update_defaults(fixed_items,"escalation",gen_inflation)
    pf_config["fixed_items"] = fixed_items

    variables = pf_config['fixed_items']
    for i in variables:
        pf.add_fixed_cost(i,variables[i]["usage"],variables[i]["unit"],variables[i]["cost"],variables[i]["escalation"])

    # ---------------------- Add feedstocks, note the various cost options-------------------
    
    pf.add_feedstock(
        name="Water",
        usage=electrolyzer_physics_results["H2_Results"]["Rated BOL: Gal H2O per kg-H2"],
        unit="gal",
        cost=feedstock_region,
        escalation=gen_inflation,
    )

    # if greenheart_config["project_parameters"]["grid_connection"]:

    energy_purchase = (
        365 * 24 * greenheart_config["electrolyzer"]["rating"] * 1e3
        + total_accessory_power_renewable_kw
        + total_accessory_power_grid_kw
    )

    pf.add_fixed_cost(
        name="Electricity from grid",
        usage=1.0,
        unit="$/year",
        cost=energy_purchase * greenheart_config["project_parameters"]["ppa_price"],
        escalation=gen_inflation,
    )

    sol = pf.solve_price()

    lcoh = sol["price"]
    if verbose:
        print("\nLCOH grid only: ", "%.2f" % (lcoh), "$/kg")
        print("ProFAST grid only NPV: ", "%.2f" % (sol["NPV"]))
        print("ProFAST grid only IRR: ", "%.5f" % (max(sol["irr"])))
        print("ProFAST grid only LCO: ", "%.2f" % (sol["lco"]), "$/kg")
        print("ProFAST grid only Profit Index: ", "%.2f" % (sol["profit index"]))
        print("ProFAST grid only payback period: ", sol["investor payback period"])

    if save_plots or show_plots:
        savepaths = [
            output_dir + "figures/capex/",
            output_dir + "figures/annual_cash_flow/",
            output_dir + "figures/lcoh_breakdown/",
            output_dir + "data/",
        ]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        pf.plot_capital_expenses(
            fileout=savepaths[0]
            + "capital_expense_grid_only_%i.pdf" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout=savepaths[1]
            + "cash_flow_grid_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )

        pd.DataFrame.from_dict(data=pf.cash_flow_out, orient="index").to_csv(
            savepaths[3] + "cash_flow_grid_only_%i.csv" % (design_scenario["id"])
        )

        pf.plot_costs(
            savepaths[2] + "lcoh_grid_only_%i" % (design_scenario["id"]),
            show_plot=show_plots,
        )
    return lcoh, pf


def run_profast_full_plant_model(
    greenheart_config,
    wind_cost_results,
    electrolyzer_physics_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    incentive_option,
    design_scenario,
    total_accessory_power_renewable_kw,
    total_accessory_power_grid_kw,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
):  
    
    gen_inflation = greenheart_config["finance_parameters"]["profast_general_inflation"]

    if "feedstock_region" in greenheart_config["site"]:
        feedstock_region = greenheart_config["site"]["feedstock_region"]
    else:
        feedstock_region = "US Average"

    if "profast_config" in greenheart_config["finance_parameters"]:
        pf_config = greenheart_config["finance_parameters"]["profast_config"]
        analysis_start_year = pf_config["params"]["analysis start year"]
        installation_period_months = pf_config["params"]["installation months"]
        # pf_desc = "Using_PFConfig"
    else:
        # pf_desc = "Using_financeparams"
        pf_config = {"params":{}}
        analysis_start_year = greenheart_config["project_parameters"]["atb_year"] + 2
        installation_period_months = wind_cost_results.installation_time
        pf_config["params"].update({"analysis start year":analysis_start_year})
        pf_config["params"].update({"installation months": installation_period_months})
        pf_config["params"].update({"sales tax":greenheart_config["finance_parameters"]["sales_tax_rate"]})
        pf_config["params"].update({"property tax and insurance":greenheart_config["finance_parameters"]["property_tax"] + 
        greenheart_config["finance_parameters"]["property_insurance"]})
        pf_config["params"].update({"admin expense":greenheart_config["finance_parameters"]["administrative_expense_percent_of_sales"]})
        pf_config["params"].update({"total income tax rate": greenheart_config["finance_parameters"]["total_income_tax_rate"]})
        pf_config["params"].update({"capital gains tax rate":greenheart_config["finance_parameters"]["capital_gains_tax_rate"]})
        pf_config["params"].update({"debt equity ratio of initial financing":greenheart_config["finance_parameters"]["debt_equity_ratio"]})
        pf_config["params"].update({"debt type": greenheart_config["finance_parameters"]["debt_type"]})
        
        pf_config["params"].update({"loan period if used": greenheart_config["finance_parameters"]["loan_period"]})
        pf_config["params"].update({"debt interest rate":greenheart_config["finance_parameters"]["debt_interest_rate"]})
        pf_config["params"].update({"cash onhand":greenheart_config["finance_parameters"]["cash_onhand_months"]})
        pf_config["params"].update({"leverage after tax nominal discount rate":greenheart_config["finance_parameters"]["discount_rate"]})

        pf_config["params"].update({"installation cost":{
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False}})
    years_of_operation = cost_tools.create_years_of_operation(greenheart_config["project_parameters"]["project_lifetime"],analysis_start_year,installation_period_months)
    electrolyzer_cost_info = summarize_electrolysis_cost_and_performance(electrolyzer_physics_results,greenheart_config["electrolyzer"])
    pf_config["params"].update({"general inflation rate": gen_inflation})
    

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        if 'land_cost' in greenheart_config['finance_parameters']:
            land_cost = greenheart_config['finance_parameters']['land_cost']
        else:
            land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf_config["params"]["commodity"] = {"name": "Hydrogen","unit": "kg","initial price": 100,"escalation":gen_inflation}
    pf_config["params"]["capacity"] = electrolyzer_cost_info["electrolyzer_capacity_kg_pr_day"]
    pf_config["params"]["operating life"] = greenheart_config["project_parameters"]["project_lifetime"]
    utilization = dict(zip(years_of_operation,electrolyzer_cost_info["electrolyzer_utilization"]))
    pf_config["params"]["long term utilization"] = utilization  # TODO should use utilization
    
    pf_config["params"]["non depr assets"] = land_cost
    pf_config["params"]["end of proj sale non depr assets"] = land_cost * (1 + gen_inflation)** greenheart_config["project_parameters"]["project_lifetime"]
    
    pf = ProFAST.ProFAST()
    params = pf_config['params']
    for i in params:
        pf.set_params(i,params[i])
    
   # ----------------------------------- Add capital items to ProFAST ----------------
    depr_type=greenheart_config["finance_parameters"]["depreciation_method"]
    depr_period=greenheart_config["finance_parameters"]["depreciation_period"]
    capital_items = {}
    for item in capex_breakdown.keys():
        if capex_breakdown[item]>0:
            if item == "electrolyzer":
                refurb = electrolyzer_cost_info["refurb_cost_simple"]
            else:
                refurb = [0]
            capital_item = cost_tools.make_profast_capital_item(capex_breakdown[item],item,refurb=refurb)
            capital_items.update(capital_item)
    capital_items = pf_tools.update_defaults(capital_items,'depr_type',depr_type)
    capital_items = pf_tools.update_defaults(capital_items,"depr_period",depr_period)
    
    pf_config["capital_items"] = capital_items

    variables = pf_config['capital_items']
    for i in variables:
        pf.add_capital_item(i,variables[i]["cost"],variables[i]["depr_type"],variables[i]["depr_period"],variables[i]["refurb"])
    
    # ----------------------------------- Add fixed items to ProFAST ----------------
    fixed_items = {}
    for item in opex_breakdown.keys():
        if opex_breakdown[item]>0:
            fixed_item = cost_tools.make_profast_fixed_cost_item(opex_breakdown[item],item)
            fixed_items.update(fixed_item)
    fixed_items = pf_tools.update_defaults(fixed_items,"escalation",gen_inflation)
    pf_config["fixed_items"] = fixed_items

    variables = pf_config['fixed_items']
    for i in variables:
        pf.add_fixed_cost(i,variables[i]["usage"],variables[i]["unit"],variables[i]["cost"],variables[i]["escalation"])
    
    if isinstance(electrolyzer_cost_info["electrolyzer_var_om"],(list,np.ndarray)):
        vopex_elec = dict(zip(years_of_operation,electrolyzer_cost_info["electrolyzer_var_om"]))
    elif isinstance(electrolyzer_cost_info["electrolyzer_var_om"],float) and (electrolyzer_cost_info["electrolyzer_var_om"]>0):
        vopex_elec = electrolyzer_cost_info["electrolyzer_var_om"] #$/kg-year
    else:
        vopex_elec = 0
    
    pf.add_feedstock(
        name = "Electrolyzer Variable O&M",
        usage = 1.0,
        unit = "$/kg",
        cost = vopex_elec,
        escalation = gen_inflation
        )


    # ---------------------- Add feedstocks, note the various cost options-------------------
    if design_scenario["electrolyzer_location"] == "onshore":
        # NOTE: why only use water feedstock for onshore? Water is free offshore?
        pf.add_feedstock(
            name="Water",
            usage=electrolyzer_physics_results["H2_Results"]["Rated BOL: Gal H2O per kg-H2"],
            unit="gal",
            cost=feedstock_region,
            escalation=gen_inflation,
        )

    if (
        greenheart_config["project_parameters"]["grid_connection"]
        or total_accessory_power_grid_kw > 0
    ):

        energy_purchase = total_accessory_power_grid_kw * 365 * 24

        if greenheart_config["project_parameters"]["grid_connection"]:
            annual_energy_shortfall = np.sum(hopp_results["energy_shortfall_hopp"])
            energy_purchase += annual_energy_shortfall

        pf.add_fixed_cost(
            name="Electricity from grid",
            usage=1.0,
            unit="$/year",
            cost=energy_purchase * greenheart_config["project_parameters"]["ppa_price"],
            escalation=gen_inflation,
        )

    # ------------------------------------- add incentives -----------------------------------
    """ Note: units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""
    sunset_years = 10
    # catch incentive option and add relevant incentives
    incentive_dict = greenheart_config["policy_parameters"][
        "option%s" % (incentive_option)
    ]

    # add wind_itc (% of wind capex)
    electricity_itc_value_dollars = 0
    electricity_itc_value_percent_re_capex = incentive_dict["electricity_itc"]
    if electricity_itc_value_percent_re_capex>0:
        renewables_for_incentives = ["wind","solar","wave","electrical_export_system"]
        for item in capex_breakdown.keys():
            if any(i in item for i in renewables_for_incentives):
                electricity_itc_value_dollars += electricity_itc_value_percent_re_capex*capex_breakdown[item]
    
    # add h2_storage_itc (% of h2 storage capex)
    itc_value_percent_h2_store_capex = incentive_dict["h2_storage_itc"]
    itc_value_dollars_h2_storage = itc_value_percent_h2_store_capex * (
        capex_breakdown["h2_storage"]
    )
    total_itc_value = itc_value_dollars_h2_storage + electricity_itc_value_dollars
    
    pf.set_params(
        "one time cap inct",
        {
            "value": total_itc_value,
            "depr type": depr_type,
            "depr period": depr_period,
            "depreciable": True,
        },
    )

    # add electricity_ptc ($/kW)
    # adjust from 1992 dollars to start year
    electricity_ptc_in_dollars_per_kw = -npf.fv(
        greenheart_config['finance_parameters']['costing_general_inflation'],
        greenheart_config["project_parameters"]["atb_year"]
        + round((installation_period_months / 12))
        - greenheart_config["finance_parameters"]["discount_years"]["electricity_ptc"],
        0,
        incentive_dict["electricity_ptc"],
    )  # given in 1992 dollars but adjust for inflation
    kw_per_kg_h2 = np.mean(np.array(electrolyzer_cost_info["electrolyzer_eff_kWh_pr_kg"])[:sunset_years])
    electricity_ptc_in_dollars_per_kg_h2 = (
        electricity_ptc_in_dollars_per_kw * kw_per_kg_h2
    )
    pf.add_incentive(
        name="Electricity PTC",
        value=electricity_ptc_in_dollars_per_kg_h2,
        decay=-gen_inflation,
        sunset_years=sunset_years,
        tax_credit=True,
    )  # TODO check decay

    # add h2_ptc ($/kg)
    h2_ptc_inflation_adjusted = -npf.fv(
        greenheart_config['finance_parameters']['costing_general_inflation'], # use ATB year (cost inflation 2.5%) costing_general_inflation
        greenheart_config["project_parameters"]["atb_year"]
        + round((installation_period_months / 12))
        - greenheart_config["finance_parameters"]["discount_years"]["h2_ptc"],
        0,
        incentive_dict["h2_ptc"],
    )
    pf.add_incentive(
        name="H2 PTC",
        value=h2_ptc_inflation_adjusted,
        decay=-gen_inflation, #correct inflation
        sunset_years=sunset_years,
        tax_credit=True,
    )  # TODO check decay

    # ------------------------------------ solve and post-process -----------------------------

    sol = pf.solve_price()

    df = pf.cash_flow_out

    lcoh = sol["price"]

    if verbose:
        print("\nProFAST LCOH: ", "%.2f" % (lcoh), "$/kg")
        print("ProFAST NPV: ", "%.2f" % (sol["NPV"]))
        print("ProFAST IRR: ", "%.5f" % (max(sol["irr"])))
        print("ProFAST LCO: ", "%.2f" % (sol["lco"]), "$/kg")
        print("ProFAST Profit Index: ", "%.2f" % (sol["profit index"]))
        print("ProFAST payback period: ", sol["investor payback period"])

        MIRR = npf.mirr(
            df["Investor cash flow"],
            greenheart_config["finance_parameters"]["debt_interest_rate"],
            greenheart_config["finance_parameters"]["discount_rate"],
        )  # TODO probably ignore MIRR
        NPV = npf.npv(
            greenheart_config["finance_parameters"]["profast_general_inflation"],
            df["Investor cash flow"],
        )
        ROI = np.sum(df["Investor cash flow"]) / abs(
            np.sum(df["Investor cash flow"][df["Investor cash flow"] < 0])
        )  # ROI is not a good way of thinking about the value of the project

        # TODO project level IRR - capex and operating cash flow

        # note: hurdle rate typically 20% IRR before investing in it due to typically optimistic assumptions
        # note: negative retained earnings (keeping debt, paying down equity) - to get around it, do another line for retained earnings and watch dividends paid by the rpoject (net income/equity should stay positive this way)

        print("Investor NPV: ", np.round(NPV * 1e-6, 2), "M USD")
        print("Investor MIRR: ", np.round(MIRR, 5), "")
        print("Investor ROI: ", np.round(ROI, 5), "")

    if save_plots or show_plots:
        savepaths = [
            output_dir + "figures/capex/",
            output_dir + "figures/annual_cash_flow/",
            output_dir + "figures/lcoh_breakdown/",
            output_dir + "data/",
        ]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        pf.plot_capital_expenses(
            fileout=savepaths[0] + "capital_expense_%i.pdf" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout=savepaths[1] + "cash_flow_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )

        pd.DataFrame.from_dict(data=pf.cash_flow_out).to_csv(
            savepaths[3] + "cash_flow_%i.csv" % (design_scenario["id"])
        )

        pf.plot_costs(
            savepaths[2] + "lcoh_%i" % (design_scenario["id"]), show_plot=show_plots,
        )

    return lcoh, pf
