import numpy as np
import os.path
from pathlib import Path
import pandas as pd


def site_details_creator(desired_lats, desired_lons, year="2012", not_rect=False):
    """
    Creates a "site_details" dataframe for analyzing
    :return: all_sites Dataframe of site_num, lat, lon, solar_filenames, wind_filenames
    """

    if type(desired_lats) == int or type(desired_lats) == float:
        N_lat = 1
    else:
        N_lat = len(desired_lats)

    if type(desired_lons) == int or type(desired_lons) == float:
        N_lon = 1
    else:
        N_lon = len(desired_lons)

    # Check if making rectilinear grid
    if not_rect:
        if N_lat != N_lon:
            raise ValueError("# of lats & longs must be the same if not making rectilinear grid")

    count = 0
    if not_rect:
        desired_lons_grid = np.zeros(N_lat)
        desired_lats_grid = np.zeros(N_lat)
    else:    
        desired_lons_grid = np.zeros(N_lat * N_lon)
        desired_lats_grid = np.zeros(N_lat * N_lon)
    if N_lat * N_lon == 1:
        desired_lats_grid = [desired_lats]
        desired_lons_grid = [desired_lons]
    else:
        for desired_lon in desired_lons:
            if not_rect:
                desired_lons_grid[count] = desired_lon
                desired_lats_grid[count] = desired_lats[count]
                count = count + 1
            else:
                for desired_lat in desired_lats:
                    desired_lons_grid[count] = desired_lon
                    desired_lats_grid[count] = desired_lat
                    count = count + 1
    site_nums = np.linspace(1, count, count)
    site_nums = site_nums.astype(int)

    all_sites = pd.DataFrame(
        {'site_nums': site_nums, 'lat': desired_lats_grid[:len(desired_lats_grid)],
         'lon': desired_lons_grid[:len(desired_lons_grid)]})

    # Fill the wind and solar resource locations with blanks (for resource API use)
    solar_filenames = []
    wind_filenames = []
    years = []
    for i in range(len(all_sites)):
        solar_filenames.append('')
        wind_filenames.append('')
        years.append(year)

    all_sites['solar_filenames'] = solar_filenames
    all_sites['wind_filenames'] = wind_filenames
    all_sites['year'] = years

    return all_sites


