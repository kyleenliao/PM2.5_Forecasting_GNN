import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

import numpy as np
import pandas as pd
import math
import metpy.calc as mpcalc
from metpy.units import units
import arrow
from datetime import datetime, timedelta
import geopy.distance
import pickle

wind_u_fp = os.path.join(proj_dir, 'data/wind_u_10_grid.npy')
wind_v_fp = os.path.join(proj_dir, 'data/wind_v_10_grid.npy')
lat_wind_fp = os.path.join(proj_dir,'data/lat_wind.npy')
lon_wind_fp = os.path.join(proj_dir,'data/lon_wind.npy')

frp_dict_fp = os.path.join(proj_dir, 'data/frp_dict.pkl')
location_fp = os.path.join(proj_dir, 'data/latlon.csv')
time_dict_fp = os.path.join(proj_dir, 'data/time_dict.pkl')
dataset_fp = os.path.join(proj_dir, 'data/dataset_fire_wind_aligned.npy')
alldates_fp = os.path.join(proj_dir, 'data/alltimes_pst.npy')
grid_dict_lat_fp = os.path.join(proj_dir, 'data/dict_wind_grid_lat.pkl')
grid_dict_lon_fp = os.path.join(proj_dir, 'data/dict_wind_grid_lon.pkl')

def getdistance(lat1, lon1, lat2, lon2):
  coords_1 = (lat1, lon1)
  coords_2 = (lat2, lon2)
  return (geopy.distance.distance(coords_1, coords_2).km)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def wind_uv_to_dir(U,V):
    """
    Calculates the wind direction from the u and v component of wind.
    Takes into account the wind direction coordinates is different than the
    trig unit circle coordinate. If the wind directin is 360 then returns zero
    (by %360)
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WDIR = mpcalc.wind_direction(U, V, convention='to')%360
    return WDIR

    
def wind_uv_to_spd(U,V):
    """
    Calculates the wind speed from the u and v wind components
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WSPD = mpcalc.wind_speed(U, V)
    return WSPD                  

def main():
	wu, wv = np.load(wind_u_fp), np.load(wind_v_fp)
	time_dict = pickle.load( open(time_dict_fp, "rb" ) )
	lat_grid, lon_grid = np.load(lat_wind_fp), np.load(lon_wind_fp) 
	frp_dict = pickle.load( open(frp_dict_fp, "rb" ) )
 
	caldorfire_lat, caldorfire_lon = 38.586, -120.537833

	grid_dict_lat = pickle.load( open(grid_dict_lat_fp, "rb" ) )
	grid_dict_lon = pickle.load( open(grid_dict_lon_fp, "rb" ) )
    
	siteloc = np.asarray( pd.read_csv(location_fp).drop("Unnamed: 0", axis=1) )

	alldates = np.arange(datetime(2017,1,1,0), datetime(2022,1,1, 0), timedelta(hours=1)).astype(datetime)
	for i in range(len(alldates)):
		alldates[i] = alldates[i].strftime('%Y-%m-%d %H')
    
	start = time_dict["2021-01-01 00:00"] 
	dataset = np.load(dataset_fp)[start:,:,:]
	alldates = np.load(alldates_fp)[start:]
    
	for i in range(dataset.shape[0]-7):
		if time_dict[alldates[i][0:13]] <= time_dict["2021-05-31 00"]:
			continue
		if i+7 >= dataset.shape[0]:
			continue
		if alldates[i+7][0:10] not in frp_dict.keys():
			continue
            
		for j in range(len(siteloc[0])): # loop over locations
			latsite, lonsite = siteloc[0][j], siteloc[1][j]

			for fire in frp_dict[alldates[i+7][0:10]]: # +7 to adjust for time difference; frp_dict in UTC, dataset in PST
				latf, lonf, frp = fire[0], fire[1], fire[2]

				timeind = time_dict[alldates[i][0:13]] # wind values collected at actual time, not simulated
				latind = grid_dict_lat[find_nearest(lat_grid, latf)]
				lonind = grid_dict_lon[find_nearest(lon_grid, lonf)]

				firewindu = wu[timeind][latind][lonind] * units.meter / units.second
				firewindv = wv[timeind][latind][lonind] * units.meter / units.second

				# get angle between fire location and site location
				bearingangle = calculate_initial_compass_bearing((latf, lonf), (latsite, lonsite))
				# get angle of wind u and v component at fire location
				wind_fire_angle = wind_uv_to_dir(firewindu, firewindv)._magnitude
                    
				if abs(bearingangle-wind_fire_angle)<90:
					weight1 = wind_uv_to_spd(firewindu, firewindv)._magnitude
					dist = getdistance(latf, lonf, latsite, lonsite)
					dist_to_prescfire = getdistance(latf, lonf, caldorfire_lat, caldorfire_lon)
					weight2 = 1/ (dist * dist * 4 * math.pi)
					weight3 = math.cos( math.radians( abs(bearingangle-wind_fire_angle) ) )
                    
					if dist_to_prescfire <= 25:
						if dist <=25:
							dataset[i,j,9] -= frp*weight1*weight2*weight3 # FRP 25KM
						if dist <=50:
							dataset[i,j,10] -= frp*weight1*weight2*weight3 # FRP 50KM
						if dist <=100:
							dataset[i,j,11] -= frp*weight1*weight2*weight3 # FRP 100KM
						if dist <=500:
							dataset[i,j,12] -= frp*weight1*weight2*weight3 # FRP 500KM
							dataset[i,j,13] -= 1 # Number of fires within 500KM

	dataset  = dataset[0:-7, :, :]
	np.save(os.path.join(proj_dir, 'data/dataset_exclude_caldor.npy'), dataset)
      
if __name__ == '__main__':
    main()
