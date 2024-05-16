import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset_exp1 import HazeData # for Experiment 1
# from dataset import HazeData # for Experiment 2

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
import numpy as np

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph()
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
seq_len = hist_len + pred_len
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')

in_dim = test_data.feature.shape[-1] + test_data.pm25.shape[-1]
wind_mean_fp, wind_std_fp = os.path.join(proj_dir, 'data/train_wind_mean.npy'), os.path.join(proj_dir, 'data/train_wind_std.npy')
wind_mean, wind_std = np.load(wind_mean_fp), np.load(wind_std_fp) 
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std 
feature_mean, feature_std = test_data.feature_mean, test_data.feature_std 

def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far

def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    else:
        raise Exception('Wrong model name!')


def prepare(pm25, feature, time_arr, flag):
    pm25_hist = np.full((pm25.shape[0], hist_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float)#double)
    pm25_label = np.full((pm25.shape[0], pred_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float)#double)
    feature = np.full((feature.shape[0], hist_len+pred_len, feature.shape[1], feature.shape[2]), -1.0, dtype=np.float)#double)
    pm = np.full((pm25.shape[0], hist_len+pred_len, pm25.shape[1], pm25.shape[2]), -1.0, dtype=np.float)#double)

    for i in range(np.asarray(time_arr).shape[0]):
        if flag == "Train":
            end = train_data.time_index[np.asarray(time_arr)[i]]
            pm25_hist[i,:,:,:] = train_data.pm25_full[end-seq_len+1:end-pred_len+1, :, :]
            pm25_label[i,:,:,:] = train_data.pm25_full[end-pred_len+1:end+1, :, :]
            feature[i,:,:,:] = train_data.feature_full[end-seq_len+1:end+1, :, :]
            pm[i,:,:,:] = train_data.pm25_full[end-seq_len+1:end+1, :, :]
        elif flag == "Val":
            end = val_data.time_index[np.asarray(time_arr)[i]]
            pm25_hist[i,:,:,:] = val_data.pm25_full[end-seq_len+1:end-pred_len+1, :, :]
            pm25_label[i,:,:,:] = val_data.pm25_full[end-pred_len+1:end+1, :, :]
            feature[i,:,:,:] = val_data.feature_full[end-seq_len+1:end+1, :, :]
            pm[i,:,:,:] = val_data.pm25_full[end-seq_len+1:end+1, :, :]
        else:
            end = test_data.time_index[np.asarray(time_arr)[i]]
            pm25_hist[i,:,:,:] = test_data.pm25_full[end-seq_len+1:end-pred_len+1, :, :]
            pm25_label[i,:,:,:] = test_data.pm25_full[end-pred_len+1:end+1, :, :]
            feature[i,:,:,:] = test_data.feature_full[end-seq_len+1:end+1, :, :]
            pm[i,:,:,:] = test_data.pm25_full[end-seq_len+1:end+1, :, :]

    return torch.tensor(pm25_hist, dtype=torch.float), torch.tensor(pm25_label, dtype=torch.float), torch.tensor(feature, dtype=torch.float), pm

def test_data_saving(test_loader, model):
    model.eval()
    predict_list, label_list, time_list = [], [], []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25_hist, pm25_label, feature, pm25 = prepare(pm25, feature, time_arr, "Test")
        feature = feature.to(device)
        pm25_hist = pm25_hist.to(device)
        pm25_label = pm25_label.to(device)
        pm25_pred = model(pm25_hist, feature)

        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label = pm25 * pm25_std + pm25_mean
        predict_list.append(pm25_pred)
        label_list.append(pm25_label)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1
    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch

def test(test_loader, model):
    model.eval()
    predict_list, label_list, time_list = [], [], []
    test_loss = 0

    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]

        pm25_pred = model(pm25_hist.to(device), feature.to(device))

        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())
    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    model = PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std).cuda()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    print(model.eval())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 0)
    
    # Run this line for Experiment 1
    test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model) 
    
    # Run this line for Experiment 2
    #test_loss, predict_epoch, label_epoch, time_epoch = test_data_saving(test_loader, model) 
    
    exp_model_dir = os.path.join("simulate_results", str(arrow.now().format('YYYYMMDDHHmmss')))
    if not os.path.exists(exp_model_dir):
        os.makedirs(exp_model_dir)
    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)
    print(exp_model_dir)

if __name__ == '__main__':
    main()
