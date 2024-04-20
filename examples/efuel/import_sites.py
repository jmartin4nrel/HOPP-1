import numpy as np
import pandas as pd
from pathlib import Path

def import_sites(fn):

    site_df = pd.read_csv(fn,index_col='PlantID')

    lats = site_df['Latitude'].values
    lons = site_df['Longitude'].values
    states = site_df['PlantState'].values

    survey_rad = 50 # survey radius in km

    earth_rad = 6371 # Earth's radium in km, needed for lat/long calcs

    lat_array = np.zeros([len(lats),19])
    lon_array = np.zeros([len(lats),19])
    for idx, lat in enumerate(lats):
        lon = lons[idx]
        lat_array[idx,0] = lat
        lon_array[idx,0] = lon
        
        # Add inner circle of 6 surrounding locations @ half of survey radius
        in_circle_lat_delta = [3**.5/4, 0, -(3**.5/4), -(3**.5/4), 0,  3**.5/4]
        for i, lat_delta in enumerate(in_circle_lat_delta):
            lat_delta = survey_rad/earth_rad*lat_delta
            lon_delta = ((survey_rad**2/4/earth_rad**2-lat_delta**2)/\
                            (np.cos(lat/180*np.pi)**2))**.5*180/np.pi
            lat_delta *= 180/np.pi
            if i<3: lon_delta = -lon_delta 
            lat_array[idx,1+i] = lat+lat_delta
            lon_array[idx,1+i] = lon+lon_delta

        # Add outer circle of 12 surrounding location @ full survey radius
        out_circle_lat_delta = [ 1,   3**.5/2,   .5, 0, -.5, -(3**.5/2),
                                -1, -(3**.5/2), -.5, 0,  .5,   3**.5/2]
        for i, lat_delta in enumerate(out_circle_lat_delta):
            lat_delta = survey_rad/earth_rad*lat_delta
            lon_delta = ((survey_rad**2/earth_rad**2-lat_delta**2)/\
                            (np.cos(lat/180*np.pi)**2))**.5*180/np.pi
            lat_delta *= 180/np.pi
            if i<6: lon_delta = -lon_delta 
            lat_array[idx,7+i] = lat+lat_delta
            lon_array[idx,7+i] = lon+lon_delta

    return lat_array, lon_array, states

if __name__ == '__main__':

    resource_dir = Path(__file__).parent/'inputs'
    import_sites(resource_dir/'ngcc_sites_full.csv')