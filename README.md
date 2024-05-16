# Prescribed_Fire_PM2.5_Simulation

Graph Neural Network (GNN) model from PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting: https://github.com/shuowang-ai/PM2.5-GNN/tree/7aacc6b6b9562ad2a9dad6197e6c4d73607ebdf2

This repository includes the code used to perform two experiments which apply the GNN model to determine the optimal month to conduct prescribed fires, and quantify the potential air quality trade-offs involved in conducting more prescribed fires outside the fire season.


## Dataset

Access all datasets not included in the data file from [Google Drive](https://drive.google.com/drive/folders/1JMv5cqvuq7A9DSHvChQ8R4MUhvrKY6Fe?usp=sharing)

## Requirements

```
Python 3.7.9
PyTorch 1.13.1
```

```bash
pip install -r reqs.txt
```

## Set-up and Run
**Train Model:**
open 'config.yaml'

- The following metero variables are used to train the model from the paper

```python
  metero_use: ['100m_u_component_of_wind',
               '100m_v_component_of_wind',
               '2m_dewpoint_temperature',
               '2m_temperature',
               'boundary_layer_height',
               'total_precipitation',
               'surface_pressure',
               'u_component_of_wind+950',
               'v_component_of_wind+950',
               'frp_25km_idw',
               'frp_50km_idw',
               'frp_100km_idw',
               'frp_500km_idw',
               'numfires',
               #'interp_flag',
               'julian_date',
               'time_of_day',]
```
- Set the data path accordingly

```python
filepath:
  GPU-Server:
    knowair_fp: /data/pm25gnn/data/dataset_fire_wind_aligned.npy
    results_dir: /data/pm25gnn/results
```

- Uncomment the model 

```python
#  model: MLP
#  model: LSTM
#  model: GRU
#  model: GC_LSTM
#  model: nodesFC_GRU
   model: PM25_GNN
#  model: PM25_GNN_nosub
```

run 'train.py'
```bash
python train.py
```

**Ambient Predictions:**
open 'util.py'

- Uncomment the correct line with 'config_exp1.yaml'
```python
#conf_fp = os.path.join(proj_dir, 'config.yaml')
conf_fp = os.path.join(proj_dir, 'config_ambient.yaml')
```

run 'train_ambient.py'
```bash
python train_ambient.py
```

**Experiment 1:**
open 'util.py'

- Uncomment the correct line with 'config_exp1.yaml'
```python
#conf_fp = os.path.join(proj_dir, 'config.yaml')
conf_fp = os.path.join(proj_dir, 'config_exp1.yaml')
```

open 'simulate.py'

- Uncomment the following lines:

```python
from dataset_exp1 import HazeData # for Experiment 1
# from dataset import HazeData # for Experiment 2
```

```python
# Run this line for Experiment 1
test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model) 
    
# Run this line for Experiment 2
#test_loss, predict_epoch, label_epoch, time_epoch = test_data_saving(test_loader, model)
```

run 'simulate.py':
```bash
python simulate.py
```

**Experiment 2:**
open 'config.yaml'

- Change the dataset file path

```python
filepath:
  GPU-Server:
    knowair_fp: /data/pm25gnn/data/dataset_fire_wind_aligned.npy
    results_dir: /data/pm25gnn/results
```

- 'exp2_data_pre.py' is run to combine the files 'transpose_pfire.py' and 'exclude_fires.py' into the final dataset used for simulation (dataset_caldor_sim_100x_2018pm25.npy)

open 'simulate.py'

- Uncomment the following lines:

```python
#from dataset_exp1 import HazeData # for Experiment 1
from dataset import HazeData # for Experiment 2
```

```python
# Run this line for Experiment 1
#test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model) 
    
# Run this line for Experiment 2
test_loss, predict_epoch, label_epoch, time_epoch = test_data_saving(test_loader, model)
```

run 'simulate.py':
```bash
python simulate.py
```
