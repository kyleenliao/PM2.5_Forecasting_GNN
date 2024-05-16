import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham
import pickle
import time
import requests

city_fp = os.path.join(proj_dir, 'data/locations.txt')
altitude_fp = os.path.join(proj_dir, 'data/alt.pkl')

class Graph():
    def __init__(self):
        self.dist_thres = 3
        self.alti_thres = 1200
        self.factor = 10
        self.use_altitude = True

        self.count = set()
        self.alt_dict = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index, self.edge_attr = self._gen_edges()
        if self.use_altitude:
            self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        assert os.path.isfile(altitude_fp)
        with open(altitude_fp, 'rb') as f:
            altitude = pickle.load(f)
        return altitude

    def _get_alt(self, latitude, longitude):
        alt = np.full(len(latitude), 0.0)
        for i in range(len(latitude)):
            lat = latitude[i]/self.factor
            lon = longitude[i]/self.factor
            if((lat,lon) in self.alt_dict):
                alt[i] = self.alt_dict[(lat,lon)]
                continue
            res = requests.get('https://api.opentopodata.org/v1/test-dataset?locations=' + str(lat)  + ',' + str(lon)).json()
            x = res["results"][0]["elevation"]
            alt[i] = (float)(x)
            time.sleep(1.0)
        return alt

    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(city_fp, 'r') as f:
            for line in f:
                idx, city, lat, lon = line.rstrip('\n').split(' ')
                idx = int(idx)
                lon = np.full(1, int(float(lon)*self.factor))
                lat = np.full(1, int(float(lat)*self.factor))
                if((lat[0]/self.factor, lon[0]/self.factor) not in self.alt_dict.keys()):
                    altitude = self._get_alt(lat, lon)
                    altitude = altitude[0]
                    self.alt_dict[(lat[0]/self.factor, lon[0]/self.factor)] = altitude
                else:
                    altitude = self.alt_dict[(lat[0]/self.factor, lon[0]/self.factor)]
                self.count.add((lat[0]/self.factor, lon[0]/self.factor))
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon[0]/self.factor, 'lat': lat[0]/self.factor}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):
        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []

        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon =(int)(self.nodes[src]['lat']*self.factor),(int)(self.nodes[src]['lon']*self.factor)
            dest_lat, dest_lon = (int)(self.nodes[dest]['lat']*self.factor), (int)(self.nodes[dest]['lon']*self.factor)
            points = np.asarray(list(bresenham(src_lat, src_lon, dest_lat, dest_lon))).transpose((1,0))

            altitude_points = self._get_alt(points[0], points[1])
            for j in range(len(points[0])):
                self.count.add((points[0][j]/self.factor, points[1][j]/self.factor))
                self.alt_dict[(points[0][j]/self.factor, points[1][j]/self.factor)] = altitude_points[j]

            altitude_src = self._get_alt(np.full(1, src_lat), np.full(1, src_lon))
            self.alt_dict[(src_lat/self.factor, src_lon/self.factor)] = altitude_src[0]
            self.count.add((src_lat/self.factor, src_lon/self.factor))

            altitude_dest = self._get_alt(np.full(1, dest_lat), np.full(1, dest_lon))
            self.alt_dict[(dest_lat/self.factor, dest_lon/self.factor)] = altitude_dest[0]
            self.count.add((dest_lat/self.factor, dest_lon/self.factor))

            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        #with open('data/alt.pkl', 'wb') as handle:
            #pickle.dump(self.alt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #with open('data/alt.pkl', 'rb') as handle:
            #b = pickle.load(handle)

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)

if __name__ == '__main__':
   graph = Graph()
