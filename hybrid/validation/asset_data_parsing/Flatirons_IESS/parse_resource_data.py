def parse_pv(res_path, years):

    filepaths = {}
    for year in years:
        filepaths[year] = res_path/'solar_m2_{}.csv'.format(year)

    return filepaths

def parse_wind(res_path, years):

    filepaths = {}
    for year in years:
        filepaths[year] = res_path/'wind_m5_{}.srw'.format(year)

    return filepaths