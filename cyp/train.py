import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool
from tqdm import tqdm
from yaml import load, dump, FullLoader

from cyp.data import CYPDataset


class Net(torch.nn.Module):
    def __init__(self, hidden_size, num_node_features, num_classes, num_conv_layers=3, num_linear_layers=1, dropout=0.5,
                 conv_layer='GCN', skip_connections=False, batch_norm=True, dummy_size=0, device=None):
        super(Net, self).__init__()
        if skip_connections:
            self.projection = torch.nn.Linear(num_node_features, hidden_size)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections
        self.dummy_size = dummy_size

        conv_layer = GATConv if conv_layer == 'GAT' else GINConv if conv_layer == 'GIN' else GCNConv

        conv_layers = [conv_layer(num_node_features, hidden_size)]
        if self.dummy_size:
            dummy_layers = [torch.nn.Linear(self.dummy_size, self.dummy_size)]
            dummy_layers_graph = [torch.nn.Linear(num_node_features, self.dummy_size)]
        if self.batch_norm:
            batch_norm_layers = [torch.nn.BatchNorm1d(hidden_size)]
        for i in range(1, num_conv_layers):
            if i == num_conv_layers - 1 and not num_linear_layers:
                if self.dummy_size:
                    dummy_layers.append(torch.nn.Linear(self.dummy_size, num_classes))
                    dummy_layers_graph.append(torch.nn.Linear(hidden_size, num_classes))
                conv_layers.append(conv_layer(hidden_size, num_classes))
            else:
                if self.dummy_size:
                    dummy_layers.append(torch.nn.Linear(self.dummy_size, self.dummy_size))
                    dummy_layers_graph.append(torch.nn.Linear(hidden_size, self.dummy_size))
                conv_layers.append(conv_layer(hidden_size, hidden_size))
                if self.batch_norm:
                    batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_size))
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        if self.dummy_size:
            self.dummy_layers = torch.nn.ModuleList(dummy_layers)
            self.dummy_layers_graph = torch.nn.ModuleList(dummy_layers_graph)
        if self.batch_norm:
            self.batch_norm_layers = torch.nn.ModuleList(batch_norm_layers)

        linear_layers = []
        if num_linear_layers:
            for i in range(num_linear_layers):
                input_size = hidden_size + self.dummy_size if i == 0 else hidden_size
                if i == num_linear_layers - 1:
                    linear_layers.append(torch.nn.Linear(input_size, num_classes))
                else:
                    linear_layers.append(torch.nn.Linear(input_size, hidden_size))
        self.linear_layers = torch.nn.ModuleList(linear_layers)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.dummy_size:
            batch_size = batch.max().item() + 1
            dummy_node = torch.zeros(batch_size, self.dummy_size).to(self.device)

        for i, conv in enumerate(self.conv_layers[:-1]):
            if self.dummy_size:
                gap = global_mean_pool(x, batch)
                dummy_node = self.dummy_layers_graph[i](gap) + self.dummy_layers[i](dummy_node)
            x = self.forward_conv(x, edge_index, conv, bn=self.batch_norm_layers[i] if self.batch_norm else None)
        if self.dummy_size:
            gap = global_mean_pool(x, batch)
            dummy_node = self.dummy_layers_graph[-1](gap) + self.dummy_layers[-1](dummy_node)
        if self.linear_layers:
            x = self.forward_conv(x, edge_index, self.conv_layers[-1],
                                  bn=self.batch_norm_layers[-1] if self.batch_norm else None)
        else:
            x = self.conv_layers[-1](x, edge_index)
            if self.dummy_size:
                x = x + dummy_node[batch]

        for i, layer in enumerate(self.linear_layers[:-1]):
            if i == 0 and self.dummy_size:
                x = torch.cat([x, dummy_node[batch]], dim=-1)
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
        if self.linear_layers:
            if len(self.linear_layers) == 1 and self.dummy_size:
                x = torch.cat([x, dummy_node[batch]], dim=-1)
            x = self.linear_layers[-1](x)

        return x

    def forward_conv(self, x, edge_index, conv, bn=None):
        x_int = conv(x, edge_index)
        if bn:
            x_int = bn(x_int)
        x_int = F.relu(x_int)
        x_int = F.dropout(x_int, training=self.training, p=self.dropout)
        if self.skip_connections:
            if x.size(-1) != x_int.size(-1):
                x = self.projection(x)
            x = x_int + x
        else:
            x = x_int
        return x


def mse_masked_loss(y_pred, y_true):
    mask = torch.isnan(y_true)
    y_pred[mask] = 0
    y_true[mask] = 0
    return F.mse_loss(y_pred, y_true)


def sign_accuracy(y_pred, y_true):
    acc = (torch.sign(y_true) == torch.sign(y_pred))[~torch.isnan(y_true)].sum() / (~torch.isnan(y_true)).sum()
    return acc.detach().cpu().numpy()


def best_ranked_accuracy(y_pred, y_true):
    mask = np.all(~np.isnan(y_true.detach().cpu().numpy()), axis=1)
    return (np.nanargmin(y_true.detach().cpu().numpy()[mask], axis=1) == np.nanargmin(
        y_pred.detach().cpu().numpy()[mask], axis=1)).sum() / mask.sum()


def train_model(model, train_loader, test_loader, optimizer=None, n_epochs=100, early_stop_patience=20,
                model_path=None, metrics=None):
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-5)

    psuj_train_loss = []
    psuj_test_loss = []

    early_stop = early_stop_patience
    min_loss = np.inf
    best_epoch = -1
    outer_loop = tqdm(range(1, n_epochs + 1), leave=False, disable=args.disable_tqdm)
    for epoch in outer_loop:
        this_psuj_train_loss = []

        model.train()
        inner_loop = tqdm(train_loader, leave=False, disable=args.disable_tqdm)
        for data in inner_loop:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = mse_masked_loss(out, data.y)
            loss.backward()
            optimizer.step()

            inner_loop.set_description(f"Train loss: {loss:.4f}")
            inner_loop.refresh()
            this_psuj_train_loss.append(loss.detach().numpy())
        psuj_train_loss.append(this_psuj_train_loss)

        loss = 0
        metric_values = {metric: 0 for metric in metrics.keys()}
        outputs = []
        num_graphs, num_nodes = 0, 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                outputs.append(out)
                loss += mse_masked_loss(out, data.y) * data.num_graphs
                for metric, (metric_fn, metric_level) in metrics.items():
                    metric_values[metric] += metric_fn(out, data.y) * (
                        data.num_graphs if metric_level == 'graph' else data.num_nodes)
                num_graphs += data.num_graphs
                num_nodes += data.num_nodes
        loss /= num_graphs
        psuj_test_loss.append(loss.detach().numpy())
        for metric, (metric_fn, metric_level) in metrics.items():
            metric_values[metric] /= (num_graphs if metric_level == 'graph' else num_nodes)

        early_stop -= 1
        if loss < min_loss:
            torch.save(model.state_dict(), model_path)
            np.save(model_path.replace('.p', '.npy'), np.concatenate([o.detach().cpu().numpy() for o in outputs]))
            min_loss = loss
            best_epoch = epoch
            best_metrics = metric_values
            early_stop = early_stop_patience
        if not early_stop:
            break

        outer_loop.set_description(f"Test loss: {loss:.4f}")
        outer_loop.refresh()

        scheduler.step(loss)

    np.save(os.path.join(args.results_path, experiment_tag, 'train_history.npy'), psuj_train_loss)
    np.save(os.path.join(args.results_path, experiment_tag, 'test_history.npy'), psuj_test_loss)
    return min_loss, best_epoch, best_metrics


def convert_config(config):
    hyperparameters = {key: value for key, value in config.items() if key != 'optimizer' and key != 'model'}
    hyperparameters.update({f'model__{key}': value for key, value in config['model'].items()})
    hyperparameters.update({f'optimizer__{key}': value for key, value in config['optimizer'].items()})
    for key, value in hyperparameters.items():
        if not isinstance(value, list):
            hyperparameters[key] = [value]
    return ParameterGrid(hyperparameters)


def convert_hyperparameters(hyperparameters):
    grid = {key: value for key, value in hyperparameters.items() if '__' not in key}
    grid['model'] = {key.split('__', 1)[1]: value for key, value in hyperparameters.items() if 'model__' in key}
    grid['optimizer'] = {key.split('__', 1)[1]: value for key, value in hyperparameters.items() if 'optimizer__' in key}
    return grid


def parse_args():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--cyp', type=str, default='CYP3A4_1W0E')
    parser.add_argument('--data-path', type=str, default='./data/molecules')
    # model arguments
    parser.add_argument('--grid-file', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-conv-layers', type=int, default=3)
    parser.add_argument('--num-linear-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--conv-layer', type=str, default='GCN')
    parser.add_argument('--skip-connections', action='store_true')
    # training arguments
    parser.add_argument('--cross-validation', action="store_true")
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--model-path', type=str, default='./models')
    parser.add_argument('--results-path', type=str, default='./results')
    parser.add_argument('--disable-tqdm', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--cyp', type=str, default='CYP3A4_1W0E')
    parser.add_argument('--data-path', type=str, default='./data')
    # model arguments
    parser.add_argument('--grid-file', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-conv-layers', type=int, default=3)
    parser.add_argument('--num-linear-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--conv-layer', type=str, default='GCN')
    parser.add_argument('--skip-connections', action='store_true')
    parser.add_argument('--dummy-size', type=int, default=128)
    # training arguments
    parser.add_argument('--cross-validation', action="store_true")
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--model-path', type=str, default='./models')
    parser.add_argument('--results-path', type=str, default='./results')
    parser.add_argument('--disable-tqdm', action='store_true')
    args = parser.parse_args()

    dataset = CYPDataset(root=args.data_path, cyp=args.cyp)

    now = datetime.datetime.now()
    experiment_tag = now.strftime("%Y%m%d_%H%M%S") + '_' + args.cyp
    output_path = os.path.join(args.results_path, experiment_tag, 'grid.csv')
    model_path = os.path.join(args.model_path, experiment_tag)
    if not os.path.exists(os.path.join(args.results_path, experiment_tag)):
        os.makedirs(os.path.join(args.results_path, experiment_tag))

    if args.grid_file:
        with open(args.grid_file, 'r') as file:
            config = load(file, Loader=FullLoader)
    else:
        config = {
            'model': {
                'num_node_features': dataset.num_node_features,
                'num_classes': dataset.num_classes,
                'hidden_size': args.hidden_size,
                'num_conv_layers': args.num_conv_layers,
                'num_linear_layers': args.num_linear_layers,
                'dropout': args.dropout,
                'conv_layer': args.conv_layer,
                'skip_connections': args.skip_connections,
                'dummy_size': args.dummy_size
            },
            'optimizer': {
                'lr': args.lr,
                'weight_decay': args.weight_decay
            },
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    grid = convert_config(config)
    if args.cross_validation:
        folds = np.array([args.n_folds * i // len(dataset) for i in range(len(dataset))])
        splits = [(dataset[np.where(folds != i)[0].tolist()], dataset[np.where(folds == i)[0].tolist()])
                  for i in range(args.n_folds)]
    else:
        splits = [(
            dataset[:int(len(dataset) * (1 - args.test_size))],
            dataset[int(len(dataset) * (1 - args.test_size)):]
        )]
        print('Split size:', len(splits[0][0]), len(splits[0][1]))

    for i, (train_dataset, test_dataset) in tqdm(enumerate(splits), desc='Folds', total=len(splits),
                                                 disable=args.disable_tqdm):
        for param_idx, params in tqdm(enumerate(grid), desc='Grid', leave=False, total=len(grid),
                                      disable=args.disable_tqdm):
            hyperparameters = convert_hyperparameters(params)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using {device}")
            model = Net(num_node_features=dataset.num_node_features, num_classes=dataset.num_classes,
                        **hyperparameters['model']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), **hyperparameters['optimizer'])
            train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'],
                                      shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'],
                                     shuffle=False)

            if not os.path.exists(os.path.join(model_path, f'fold_{i}')):
                os.makedirs(os.path.join(model_path, f'fold_{i}'))
            valid_loss, epoch, metrics = train_model(model, train_loader, test_loader, optimizer,
                                                     hyperparameters['epochs'],
                                                     model_path=os.path.join(model_path, f'fold_{i}',
                                                                             f'checkpoint_{param_idx}.p'),
                                                     metrics={'sign_acc': (sign_accuracy, 'graph'),
                                                              'rank_acc': (best_ranked_accuracy, 'node')})

            with open(os.path.join(model_path, f'config_{param_idx}.yml'), 'w') as file:
                dump(hyperparameters, file)

            progress_log = {
                'valid_loss': valid_loss.item(),
                'epoch': epoch,
                'fold': i,
                'hyperparameters_idx': param_idx
            }
            progress_log.update(params)
            progress_log.update(metrics)
            pd.DataFrame(data=progress_log, index=[0]).to_csv(output_path, mode='a',
                                                              header=not os.path.exists(output_path))
