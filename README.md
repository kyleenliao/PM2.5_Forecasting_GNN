# PM2.5_Forecasting_GNN
Simulating the Air Quality Impact of Prescribed Fires Using Graph Neural Network-Based PM2.5 Forecasts

This repository includes the code used to train the PM2.5-GNN model from Wang et al. (2020) to predict the hourly PM2.5 in California, estimate the ambient PM2.5 pollution, and predict the PM2.5 during simulated prescribed burn events. For the simulated prescribed burns, two expimerents were performed, with the first applying the PM2.5-GNN model to determine the optimal month to conduct prescribed fires, and the second quantify the potential air quality trade-offs involved in conducting more prescribed fires outside the fire season. 

Wang, S., Li, Y., Zhang, J. et al. (2020). PM2.5-GNN. Proceedings of the 28th International Conference on Advances in Geographic Information Systems. https://doi.org/10.1145/3397536.3422208

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
### Predict Hourly PM2.5 

open `config.yaml`

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

run `train.py`
```bash
python train.py
```

### Predict Ambient PM2.5

open `util.py`

- Uncomment the correct line with `config_exp1.yaml`
```python
#conf_fp = os.path.join(proj_dir, 'config.yaml')
conf_fp = os.path.join(proj_dir, 'config_ambient.yaml')
```

run `train_ambient.py`
```bash
python train_ambient.py
```

### PM2.5 Predictions during Simulated Prescribed Burn: Experiment 1

open `util.py`

- Uncomment the correct line with
```python
#conf_fp = os.path.join(proj_dir, 'config.yaml')
conf_fp = os.path.join(proj_dir, 'config_exp1.yaml')
```

open `simulate.py`

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

run `simulate.py`:
```bash
python simulate.py
```

### PM2.5 Predictions during Simulated Prescribed Burn: Experiment 2

open `config.yaml`

- Change the dataset file path

```python
filepath:
  GPU-Server:
    knowair_fp: /data/pm25gnn/data/dataset_fire_wind_aligned.npy
    results_dir: /data/pm25gnn/results
```

- `exp2_data_pre.py` is run to combine the files created by `transpose_pfire.py` and `exclude_fires.py` to form the final dataset used for simulation (dataset_caldor_sim_100x_2018pm25.npy)

open `simulate.py`

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

run `simulate.py`:
```bash
python simulate.py
```
