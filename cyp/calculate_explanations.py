import sys
import os
import os.path as osp

import yaml
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import DataLoader

# silence!
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append("/home/pocha/cyp_study")
from data import CYPDataset
from explain_utils import saliency_map_increase, saliency_map_decrease, saliency_map_both 
from explain_utils import plot_explanations, SalientExplanations, Explanations, InterpretableNet


# setting things up
cyp = "CYP2D6_2F9Q"
model_path = f"/home/pocha/shared-sin/results/danel/cyp/models/20211014_145931_{cyp}"
model_id = '46'
fold = sys.argv[1]
ciekawe_zwiazki = "CYP2D6_2F9Q_repeated2.csv"

results_path = os.path.join('/', 'home', 'pocha', 'shared-sin', 'results', 'pocha', 'cyp', 'explanations', f'{cyp}', f'{fold}')

try:
    os.makedirs(results_path)
except FileExistsError:
    pass

try:
    os.makedirs(osp.join(results_path, 'explanations'))
except FileExistsError:
    pass


# loading data
data_path = "/home/pocha/cyp_study/data2"
batch_size = 1
dataset = CYPDataset(root=data_path, cyp=cyp)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
ciekawe_zwiazki = pd.read_csv(osp.join(data_path, ciekawe_zwiazki))

# loading model
with open(osp.join(model_path, f'config_{model_id}.yml'), 'r') as f:
    mdl_cfg = yaml.safe_load(f)
    
model = InterpretableNet(num_node_features=dataset.num_node_features, num_classes=dataset.num_classes, **mdl_cfg['model'])

model.load_state_dict(
    torch.load(osp.join(model_path, fold, f'checkpoint_{model_id}.p'),
               map_location=torch.device('cuda:0')),
    strict=True)

model.eval()
model = model.cuda()

device = 'cuda:0'
optimizer = torch.optim.Adam(model.parameters(), **mdl_cfg['optimizer'])


# saving config
konfig = {'results_path': results_path,
          'cyp': cyp,
          'data_path': data_path,
          'model_path': model_path,
          'model_id': model_id,
          'fold': fold,
          'model_config': mdl_cfg,
          'model': str(model)}

with open(osp.join(results_path, 'konfig.yaml'), 'w') as f:
    yaml.safe_dump(konfig, f)


# running stuff
legend = {}
for idum, data in enumerate(data_loader):
    data = data.to(device)
    
    legend[idum] = data.smiles[0]
    
    if data.smiles[0] not in ciekawe_zwiazki.smiles.values:
        continue
    
    np.save(osp.join(results_path, f'{idum}_adj.npy'), data.edge_index.detach().cpu().numpy(), allow_pickle=False)
    np.save(osp.join(results_path, f'{idum}_f_mat.npy'), data.x.detach().cpu().numpy(), allow_pickle=False)
    
    n_atoms = data.x.shape[0]
    n_substitutions = data.y.shape[1]
    
    sal_inc = np.zeros(shape=(n_atoms, n_substitutions, n_atoms))  # atoms x substitutions x explanation
    sal_dec = np.zeros(shape=(n_atoms, n_substitutions, n_atoms))  # unsigned gradcam
    sal_all = np.zeros(shape=(n_atoms, n_substitutions, n_atoms))  # saliency
    
    for atom, substitution in torch.logical_not(torch.isnan(data.y)).nonzero():
        print(f"{[int(atom), int(substitution)]}\t{data.smiles[0]}")
        
        out = model(data)
        optimizer.zero_grad()
        model.input.grad = None
        out[atom, substitution].backward(retain_graph=True)

        try:
            sal_i = saliency_map_increase(model.input.grad, regularise=False)
            sal_d = saliency_map_decrease(model.input.grad, regularise=False)
            sal_a = saliency_map_both(model.input.grad, regularise=False)

            sal_inc[atom, substitution] = sal_i
            sal_dec[atom, substitution] = sal_d
            sal_all[atom, substitution] = sal_a

            expl = SalientExplanations(sal_i, sal_d, sal_a)

            plot_explanations(data,
                              explanations=expl,
                              storage=results_path,
                              identifier=f'{idum}-{(int(atom), int(substitution))}', 
                              show=False,
                              n_atom=atom,
                              model_an=model)
        except Exception as e:
            print(e)
            continue

    np.save(osp.join(results_path, f'{idum}_saliency.npy'),
            sal_inc, allow_pickle=False)
    np.save(osp.join(results_path, f'{idum}_saliency_negative.npy'),
            sal_dec, allow_pickle=False)
    np.save(osp.join(results_path, f'{idum}_saliency_whole.npy'),
            sal_all, allow_pickle=False)
    
pd.DataFrame.from_dict(legend, orient='index',
                       columns=['smiles']).to_csv(osp.join(results_path, "legend.csv"))
