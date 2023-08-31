import numpy as np
import pandas as pd
import math
import metpy.calc as mpcalc
from metpy.units import units
import arrow
from datetime import datetime, timedelta
import geopy.distance
import pickle

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
    #WDIR= (270-np.rad2deg(np.arctan2(V,U)))%360
    WDIR = mpcalc.wind_direction(U, V, convention='to')%360
    return WDIR

    
def wind_uv_to_spd(U,V):
    """
    Calculates the wind speed from the u and v wind components
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    #WSPD = np.sqrt(np.square(U)+np.square(V))
    WSPD = mpcalc.wind_speed(U, V)
    return WSPD  
    
def get_pfires_frp(caldorfire_lat, caldorfire_lon):
    frp_dict = pickle.load( open("data/frp_dict.pkl", "rb" ) )
    start_date = datetime(2018, 3, 21)
    end_date = datetime(2020, 12, 31)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
        
    frp_pfire = {}

    for i in date_list:
        if i in frp_dict.keys():
            for j in frp_dict[i]:
                lat = j[0]
                lon = j[1]
                if getdistance(lat,lon, caldorfire_lat, caldorfire_lon) <= 25: #38.42631890642784, -120.44328186172089) < 10:
                    if i[5:] not in frp_pfire.keys():
                        frp_pfire[i[5:]] = set()
                    frp_pfire[i[5:]].add((lat, lon, j[2], i))
                    print(i, lat, lon, j[2])    
    return frp_pfire

def main():
	wu = np.load("data/wind_u_10_grid.npy")
	wv = np.load("data/wind_v_10_grid.npy")
	frp = pd.read_csv("data/raw_FRP_combined.csv")
	time_dict = pickle.load( open("data/time_dict.pkl", "rb" ) )
              
	caldorfire_lat = 38.586
	caldorfire_lon = -120.537833

	lat_grid = [41.78, 41.53, 41.28, 41.03, 40.78, 40.53, 40.28, 40.03, 39.78, 39.53,
	       39.28, 39.03, 38.78, 38.53, 38.28, 38.03, 37.78, 37.53, 37.28, 37.03,
	       36.78, 36.53, 36.28, 36.03, 35.78, 35.53, 35.28, 35.03, 34.78, 34.53,
	       34.28, 34.03, 33.78, 33.53, 33.28, 33.03, 32.78, 32.53]
	lon_grid = [-124.41, -124.16, -123.91, -123.66, -123.41, -123.16, -122.91, -122.66,
	       -122.41, -122.16, -121.91, -121.66, -121.41, -121.16, -120.91, -120.66,
	       -120.41, -120.16, -119.91, -119.66, -119.41, -119.16, -118.91, -118.66,
	       -118.41, -118.16, -117.91, -117.66, -117.41, -117.16, -116.91, -116.66,
	       -116.41, -116.16, -115.91, -115.66, -115.41, -115.16, -114.91, -114.66,
	       -114.41, -114.16]

	griddiclat = {}
	griddiclon = {}
	for i in range(len(lat_grid)):
	  griddiclat[lat_grid[i]] = i
	for i in range(len(lon_grid)):
	  griddiclon[lon_grid[i]] = i
      
	siteloc = pd.read_csv("data/latlon.csv")
	siteloc = siteloc.drop("Unnamed: 0", axis=1)
	siteloc = np.asarray(siteloc)

	alldates = np.arange(datetime(2017,1,1,0), datetime(2022,1,1, 0), timedelta(hours=1)).astype(datetime)
	for i in range(len(alldates)):
	  alldates[i] = alldates[i].strftime('%Y-%m-%d %H')
    
	pfire_frp_dic = get_pfires_frp(caldorfire_lat, caldorfire_lon)
	dataset = np.load("data/dataset_fire_wind_aligned.npy")
	alldates = np.load("data/alltimes_pst.npy")
    
	start = 36959
	end = 38687
    
	dataset = dataset[start:end, :, :]
	alldates = alldates[start:end]
	print(alldates[0], alldates[-1]) 
	print(caldorfire_lat, caldorfire_lon)

	for i in range(dataset.shape[0]-7):
		if alldates[i+7][5:10] not in pfire_frp_dic.keys():
			continue
		np.save("data/dataset_pfire_transpose_100x.npy", dataset)
            
		for j in range(len(siteloc[0])): # loop over locations
			latsite = siteloc[0][j]
			lonsite = siteloc[1][j]

			for fire in pfire_frp_dic[alldates[i+7][5:10]]: # +7 to adjust ; frp_dict needs values in UTC, but values right now are in PST
				latf = fire[0]
				lonf = fire[1]
				frp = fire[2]

				timeind = time_dict[alldates[i][0:13]] # wind values collected at actual time, not simulated
				latind = griddiclat[find_nearest(lat_grid, latf)]
				lonind = griddiclon[find_nearest(lon_grid, lonf)]

				firewindu = wu[timeind][latind][lonind] * units.meter / units.second
				firewindv = wv[timeind][latind][lonind] * units.meter / units.second

				#get angle between fire location and site location
				bearingangle = calculate_initial_compass_bearing((latf, lonf), (latsite, lonsite))
				# get angle of wind u and v component at fire location
				wind_fire_angle = wind_uv_to_dir(firewindu, firewindv)._magnitude
                    
				if abs(bearingangle-wind_fire_angle)<90:
					weight1 = wind_uv_to_spd(firewindu, firewindv)._magnitude
					dist = getdistance(latf, lonf, latsite, lonsite)
					dist_to_prescfire = getdistance(latf, lonf, caldorfire_lat, caldorfire_lon)
					weight2 = 1/ (dist * dist * 4 * math.pi)
					weight3 = math.cos( math.radians( abs(bearingangle-wind_fire_angle) ) )
					scalefactor = 100
 
					if dist_to_prescfire <= 25:
						if dist <=25:
							dataset[i,j,9] += frp*weight1*weight2*weight3*scalefactor # FRP 25KM
						if dist <=50:
							dataset[i,j,10] += frp*weight1*weight2*weight3*scalefactor # FRP 50KM
						if dist <=100:
							dataset[i,j,11] += frp*weight1*weight2*weight3*scalefactor # FRP 100KM
						if dist <=500:
							dataset[i,j,12] += frp*weight1*weight2*weight3*scalefactor # FRP 500KM
							dataset[i,j,13] += 1 # Number of fires within 500KM
	

	print(start, end)
	dataset  = dataset[0:-7, :, :]
	np.save("data/dataset_pfire_transpose_100x.npy", dataset)
      
	print("DONE")

if __name__ == '__main__':
    main()
