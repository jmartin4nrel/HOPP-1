def parse_pv(gen_path, years):

    filepaths = {}
    for year in years:
        filepaths[year] = gen_path/'FirstSolar_{}.csv'.format(year)

    return filepaths

def parse_wind(gen_path, years):

    filepaths = {}
    for year in years:
        filepaths[year] = gen_path/'GE1pt5MW_{}.csv'.format(year)

    return filepaths