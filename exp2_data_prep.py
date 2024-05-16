import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

import numpy as np
import pickle

pfire_fp = os.path.join(proj_dir,'data/dataset_pfire_transpose_100x.npy')
exclude_fp = os.path.join(proj_dir,'data/dataset_exclude_caldor.npy')
dataset_fp = os.path.join(proj_dir, 'data/dataset_fire_wind_aligned.npy')
time_dict_fp = os.path.join(proj_dir, 'data/time_dict.pkl')

pfire = np.load(pfire_fp) # created from 'transpose_pfire.py'
exclude = np.load(exclude_fp) # created from 'exclude_fires.py'
dataset = np.load(dataset_fp)
time_dict = pickle.load( open(time_dict_fp, "rb" ) )

d2018 = dataset[time_dict["2018-08-14 00:00"]:time_dict["2018-10-21 00:00"], :, -1] # contains observed PM2.5 from 2018-08-14 00:00 to 2018-10-21 00:00

# replace from 1/1/21 - 12/31/21 with caldor fire abscence data
dataset[time_dict["2021-01-01 00:00"]:time_dict["2021-01-01 00:00"]+exclude.shape[0], :, :] = exclude 

# replace from 3/21/21 - 5/31/21 with prescribed fire transposed data
dataset[time_dict["2021-03-21 00:00"]:time_dict["2021-03-21 00:00"]+pfire.shape[0], :, :] = pfire 

# replace PM2.5 values from 2021-08-14 00:00 to 2021-10-21 00:00 with observed PM2.5 from 2018-08-14 00:00 to 2018-10-21 00:00
dataset[time_dict["2021-08-14 00:00"]:time_dict["2021-10-21 00:00"], :, -1] = d2018 

np.save(os.path.join(proj_dir, 'data/dataset_caldor_sim_100x_2018pm25.npy'), dataset)
