

import numpy as np
def make_profast_capital_item(capex_usd,component,refurb=[0]):
    keys = ["cost","depr_type","depr_period","refurb"]
    vals = [capex_usd,"MACRS",7,refurb]
    name = " ".join(t.capitalize() for t in component.split("_"))
    name = name.replace("System","")
    return {"{} System".format(name):dict(zip(keys,vals))}

def make_profast_fixed_cost_item(fixed_om_usd,component):
    keys = ["usage","unit","cost","escalation"]
    vals = [1.0,"$/year",fixed_om_usd,0.0]
    name = " ".join(t.capitalize() for t in component.split("_"))
    return {"{} O&M Cost".format(name):dict(zip(keys,vals))}

def make_profast_feedstock_item(feedstock_usage_pr_kg,feedstock_name,feedstock_unit,site_feedstock_region = "US Average"):
    valid_pf_feedstocks = ["Coal","Biomass","Diesel","Water"]
    valid_pf_feedstocks += ["Electricity (commercial)","Electricity (industrial)","Electricity (solar)"]
    valid_pf_feedstocks += ["Electricity (on-shore wind)","Natural Gas (commercial)","Natural Gas (industrial)"]
    if feedstock_name in valid_pf_feedstocks:
        keys = ["usage","unit","cost","escalation"]
        if site_feedstock_region is None:
            site_feedstock_region = "US Average"
        vals = [feedstock_usage_pr_kg,feedstock_unit,site_feedstock_region,0.0]
        res = {"{}".format(feedstock_name):dict(zip(keys,vals))}
    # vals = [water_usage_gal_pr_kg,"gal",site_feedstock_region,profast_general_inflation]
    else:
        res = {}
    return res

def make_profast_variable_cost_item_h2(vopex,component):
    keys = ["usage","unit","cost","escalation"]
    vals = [1.0,"$/kg",vopex,0.0]
    name = " ".join(t.capitalize() for t in component.split("_"))
    return {"{} Variable O&M".format(name):dict(zip(keys,vals))}

def make_profast_variable_cost_item_energy(vopex,component):
    keys = ["usage","unit","cost","escalation"]
    vals = [1.0,"$/kWh",vopex,0.0]
    name = " ".join(t.capitalize() for t in component.split("_"))
    return {"{} Variable O&M".format(name):dict(zip(keys,vals))}

def make_profast_ptc_incentives(value,name,gen_inf,sunset_years):
    keys = ["value","decay","sunset_years","tax_credit"]
    vals = [value,-gen_inf,sunset_years,True]
    return {name:dict(zip(keys,vals))}

def make_profast_itc_incentives(itc_vals,depr_type,depr_perd):
    value = sum(itc_vals)
    keys = ["value","depr type","depr period","depreciable"]
    vals = [value,depr_type,depr_perd,True]
    # vals = [val,"MACRS",7 True]
    incentive_dict = dict(zip(keys,vals))
    
    return {"one time cap inct":incentive_dict}

def create_years_of_operation(plant_life_years,analysis_start_year,installation_period_months):
    operation_start_year = analysis_start_year + (installation_period_months/12)
    years_of_operation = np.arange(int(operation_start_year),int(operation_start_year+plant_life_years),1)
    year_keys = ['{}'.format(y) for y in years_of_operation]
    return year_keys