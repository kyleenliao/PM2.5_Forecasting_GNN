import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

from util import config, file_dir
import numpy as np
import pandas as pd
import math
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime, timedelta
import geopy.distance
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data
import arrow
import pdb
import pickle

wind_u_fp = os.path.join(proj_dir, 'data/wind_u_10_grid.npy')
wind_v_fp = os.path.join(proj_dir, 'data/wind_v_10_grid.npy')
grid_dict_lat_fp = os.path.join(proj_dir, 'data/dict_wind_grid_lat.pkl')
grid_dict_lon_fp = os.path.join(proj_dir, 'data/dict_wind_grid_lon.pkl')
lat_wind_fp = os.path.join(proj_dir,'data/lat_wind.npy')
lon_wind_fp = os.path.join(proj_dir,'data/lon_wind.npy')

frp_dict_fp = os.path.join(proj_dir, 'data/frp_dict.pkl')
location_fp = os.path.join(proj_dir, 'data/latlon.csv')
time_dict_fp = os.path.join(proj_dir, 'data/time_dict.pkl')

pm25_input_fp = os.path.join(proj_dir,'data/input_exp1/pm25.npy') 
feature_input_fp = os.path.join(proj_dir,'data/input_exp1/feature.npy') 
time_input_fp = os.path.join(proj_dir,'data/input_exp1/time.npy') 

test_feat_mean_fp = os.path.join(proj_dir,'data/test_feat_mean.npy') 
test_feat_std_fp = os.path.join(proj_dir,'data/test_feat_std.npy') 
test_pm25_mean_fp = os.path.join(proj_dir,'data/test_pm25_mean.npy') 
test_pm25_std_fp = os.path.join(proj_dir,'data/test_pm25_std.npy') 

class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=240,
                       pred_len=48,
                       dataset_num=1,
                       flag='Train',
                       ):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
            start_time_sim_window = 'sim_window_start'
            end_time_sim_window = 'sim_window_end'
        else:
            raise Exception('Wrong Flag!')

        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])
        self.sim_window_start = config['dataset'][dataset_num][start_time_sim_window]
        self.sim_window_end = config['dataset'][dataset_num][end_time_sim_window]
        self.window = self._get_window(self.sim_window_start, self.sim_window_end)
        
        self.wu, self.wv = np.load(wind_u_fp), np.load(wind_v_fp)
        self.time_dict = pickle.load( open(time_dict_fp, "rb" ) )
        self.frp_dict = pickle.load( open(frp_dict_fp, "rb" ) )
        self.grid_dict_lat = pickle.load( open(grid_dict_lat_fp, "rb" ) )
        self.grid_dict_lon = pickle.load( open(grid_dict_lon_fp, "rb" ) )

        self.siteloc = np.asarray( pd.read_csv(location_fp).drop("Unnamed: 0", axis=1) )
        self.lat_grid, self.lon_grid = np.load(lat_wind_fp), np.load(lon_wind_fp)

        self.knowair_fp = file_dir['knowair_fp']
        self.graph = graph
        
        # uncomment if not loading the saved inputs
        '''
        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()
        '''
        
        self.pm25, self.feature, self.time_arr = np.load(pm25_input_fp), np.load(feature_input_fp), np.load(time_input_fp) 
        
        self._calc_mean_std()
        seq_len = hist_len + pred_len

        # uncomment if not loading the saved inputs
        '''
        self._add_time_dim(seq_len)
        self._replace_pm25()
        self._recalculate_frp()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        np.save(os.path.join(proj_dir, 'data/input_exp1/feature.npy'), self.feature)
        np.save(os.path.join(proj_dir, 'data/input_exp1/pm25.npy'), self.pm25)
        np.save(os.path.join(proj_dir, 'data/input_exp1/time.npy'), self.time)        
        '''
        
        self._norm(flag)
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)

    def _replace_pm25(self):
        for i in range(self.pm25.shape[0]):
            for w in range(self.feature.shape[1]):
                 timeind = self.time_dict[self.window[w]] # pm2.5 value of simulation window
                 self.pm25[i, w, :, 0] = self.knowair[timeind, :, -1]

    def _recalculate_frp(self):
        assert self.feature.shape[1] == len(self.window)

        for i in range(self.feature.shape[0]): # loop for all time
            for w in range(self.feature.shape[1]): # loop for window length
                if self.window[w][:10] not in self.frp_dict.keys():
                    continue
                for j in range(len(self.siteloc[0])): # loop over locations
                    latsite, lonsite = self.siteloc[0][j], self.siteloc[1][j]
                    
                    for fire in self.frp_dict[self.window[w][:10]]: # loop over all fires in the simulation window 
                        latf, lonf, frp = fire[0], fire[1], fire[2]

                        timeind = self.time_dict[str(datetime.fromtimestamp(self.time_arr[i,w]))[0:13]] # wind values collected at actual time, not simulated
                        latind = self.grid_dict_lat[self._find_nearest(self.lat_grid, latf)]
                        lonind = self.grid_dict_lon[self._find_nearest(self.lon_grid, lonf)]

                        firewindu = self.wu[timeind][latind][lonind] * units.meter / units.second
                        firewindv = self.wv[timeind][latind][lonind] * units.meter / units.second

                        # get angle between fire location and site location
                        bearingangle = self._calculate_initial_compass_bearing((latf, lonf), (latsite, lonsite))
                        # get angle of wind u and v component at fire location
                        wind_fire_angle = self._wind_uv_to_dir(firewindu, firewindv)._magnitude


                        if abs(bearingangle-wind_fire_angle)<90:
                            weight1 = self._wind_uv_to_spd(firewindu, firewindv)._magnitude
                            dist = self._getdistance(latf, lonf, latsite, lonsite)
                            weight2 = 1/ (dist*dist*4*math.pi)
                            weight3 = math.cos( math.radians( abs(bearingangle-wind_fire_angle) ) )

                            if dist <=25:
                                self.feature[i,w,j,9] += frp*weight1*weight2*weight3 # FRP 25KM
                            if dist <=50:
                                self.feature[i,w,j,10] += frp*weight1*weight2*weight3 # FRP 50KM
                            if dist <=100:
                                self.feature[i,w,j,11] += frp*weight1*weight2*weight3 # FRP 100KM
                            if dist <=500:
                                self.feature[i,w,j,12] += frp*weight1*weight2*weight3 # FRP 500KM
                                self.feature[i,w,j,13] += 1 # Number of fires within 500KM

    def _norm(self, flag):
        if flag == 'Test':
            self.feature_mean = np.load(test_feat_mean_fp)
            self.feature_std = np.load(test_feat_std_fp)
            self.pm25_mean = np.load(test_pm25_mean_fp)
            self.pm25_std = np.load(test_pm25_std_fp)

        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)

        keep = np.arange(0, self.pm25.shape[0]-1, 24)
        l = self.pm25.shape[0]
        for i in range(l-1, -1, -1):
            if i not in keep:
                self.pm25 = np.delete(self.pm25, i, 0)
                self.feature = np.delete(self.feature,i,0)
                self.time_arr = np.delete(self.time_arr, i,0)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.wind_mean = self.feature_mean[7:9] # 7 and 8 index of u and v component_of_wind+950
        self.wind_std = self.feature_std[7:9] # 7 and 8 index of u and v component_of_wind+950
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _process_feature(self):
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:,:,metero_idx]

        u = self.feature[:, :, 7] * units.meter / units.second # 7 is the index of u_component_of_wind+950
        v = self.feature[:, :, 8] * units.meter / units.second # 8 is the index of v_component_of_wind+950
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]


    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        # determines time granularity (1 hour)
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+1), 1):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]

    def _get_idx(self, t):
        t0 = self.data_start
        # determines time granularity
        return int((t.timestamp - t0.timestamp) / (60 * 60))

    def _get_window(self, start, end):
        window = np.arange(datetime(start[0][0],start[0][1],start[0][2],0), datetime(end[0][0],end[0][1],end[0][2],0), timedelta(hours=1)).astype(datetime)
        for i in range(len(window)):
            window[i] = window[i].strftime('%Y-%m-%d %H')
        return window

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index]

    def _getdistance(self, lat1, lon1, lat2, lon2):
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return (geopy.distance.distance(coords_1, coords_2).km)

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    def _calculate_initial_compass_bearing(self, pointA, pointB):
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


    def _wind_uv_to_dir(self, U, V):
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


    def _wind_uv_to_spd(self, U, V):
        """
        Calculates the wind speed from the u and v wind components
        Inputs:
          U = west/east direction (wind from the west is positive, from the east is negative)
          V = south/noth direction (wind from the south is positive, from the north is negative)
        """
        #WSPD = np.sqrt(np.square(U)+np.square(V))
        WSPD = mpcalc.wind_speed(U, V)
        return WSPD

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')
