import os
import random

from collections import namedtuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from cairosvg import svg2png, svg2svg
from skimage.io import imread

from rdkit import Chem
from rdkit.Chem import rdmolops, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# silence!
import warnings
warnings.simplefilter('ignore')

import scipy

from cyp.train import Net


class InterpretableNet(Net):
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data):
        self.input, edge_index = data.x, data.edge_index
        self.input.requires_grad = True
        
        x = data.x
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = self.forward_conv(x, edge_index, conv,
                                  bn=self.batch_norm_layers[i] if self.batch_norm else None)
        
        if self.linear_layers:
            with torch.enable_grad():
                self.final_conv_acts = self.forward_conv(x, edge_index, self.conv_layers[-1],
                                                         bn=self.batch_norm_layers[-1] if self.batch_norm else None)
            self.final_conv_acts.register_hook(self.activations_hook)

            x = self.final_conv_acts
        else:
            raise RunTimeError("No nie wiem, nie wiem, jak to zrobić...")
            x = self.conv_layers[-1](x, edge_index)

        for i, layer in enumerate(self.linear_layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
        if self.linear_layers:
            x = self.linear_layers[-1](x)

        return x



def calculate_gradcams(model, regularise=False):
    final_conv_acts = model.final_conv_acts
    final_conv_grads = model.final_conv_grads
    
#     print(final_conv_grads)
#     print(final_conv_grads/27)

    grad_cam_weights = np.array(grad_cam(final_conv_acts, final_conv_grads, regularise=regularise))
    ugrad_cam_weights = np.array(ugrad_cam(final_conv_acts, final_conv_grads, regularise=regularise))
    
    return (grad_cam_weights, ugrad_cam_weights)


def grad_cam(final_conv_acts, final_conv_grads, regularise=False):
    node_heat_map = []
#     print(final_conv_grads.shape)
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
#     print(alphas)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    if regularise:
        return simple_scale(node_heat_map)
    else:
        return node_heat_map


def ugrad_cam(final_conv_acts, final_conv_grads, regularise=False):
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)

    node_heat_map = np.array(node_heat_map[:final_conv_acts.shape[0]]).reshape(-1, 1)
    
    if regularise:
        return two_way_scale(node_heat_map)
    else:
        return node_heat_map


def saliency_map_increase(input_grads, regularise=False):
    # this is the classical Saliency Map
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_saliency = torch.norm(F.relu(input_grads[n,:])).item()
        node_saliency_map.append(node_saliency)
    if regularise:
        return simple_scale(node_saliency_map)
    else:
        return node_saliency_map
    
    
def saliency_map_decrease(input_grads, regularise=False):
    # this is Saliency Map where we only look at negative gradients
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_saliency = torch.norm(F.relu(-input_grads[n,:])).item()
        node_saliency_map.append(node_saliency)
    if regularise:
        return simple_scale(node_saliency_map)
    else:
        return node_saliency_map
    
    
def saliency_map_both(input_grads, regularise=False):
    # this is a saliency maps where we look at all gradients
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_saliency = torch.norm(input_grads[n,:]).item()
        node_saliency_map.append(node_saliency)
    if regularise:
        return simple_scale(node_saliency_map)
    else:
        return node_saliency_map


def simple_scale(stuff):
    return MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(stuff).reshape(-1, 1)).reshape(-1, )


def two_way_scale(stuff):
    # te normalizacje zrobilabym inaczej...?
    pos_node_heat_map = MinMaxScaler(feature_range=(0,1)).fit_transform(stuff*(stuff >= 0)).reshape(-1,)
    neg_node_heat_map = MinMaxScaler(feature_range=(-1,0)).fit_transform(stuff*(stuff < 0)).reshape(-1,)
    return pos_node_heat_map + neg_node_heat_map


def my_scale(stuff):
    # zero w zerze, długosc 1
    scaler = np.max(stuff) - np.min(stuff)
    return (1./scaler) * stuff


Explanations = namedtuple('Explanations', ['salient', 'gradcam', 'ugradcam'])
SalientExplanations = namedtuple('SalientExplanations', ['sal_increase', 'sal_decrease', 'sal_both'])


def plot_explanations(data, model=None, explanations=None, storage='.', identifier='', show=True, n_atom=None, model_an=None):
    # if model_an is not None additional explanations will be given
    # TODO: remove model an???
    
    assert (model is not None) != (explanations is not None), 'Must provide either `model` or `explanations`.'
    
    # keeping track of naming plots
    Titles = namedtuple('Titles', ['left', 'centre', 'right'])
    
    if model is not None:
        left_weights = saliency_map_increase(model.input.grad, regularise=True)
        centre_weights, right_weights = calculate_gradcams(model, regularise=True)
        titles = Titles('Saliency Map', 'Grad-CAM', 'UGrad-CAM')
    else:
        try:
            # zawsze regularyzujemy, jak juz bylo zregularyzowane to nic sie nie stanie
            left_weights = simple_scale(explanations.salient)
            centre_weights = simple_scale(explanations.gradcam)
            right_weights = two_way_scale(explanations.ugradcam)
            titles = Titles('Saliency Map', 'Grad-CAM', 'UGrad-CAM')

        except AttributeError:
            left_weights = simple_scale(explanations.sal_increase)
            centre_weights = simple_scale(explanations.sal_decrease)
            right_weights = simple_scale(explanations.sal_both)
            titles = Titles('Saliency map', 'Saliency map (negative derivatives)', 'Saliency map (all derivatives)')
        
    mol = Chem.MolFromSmiles(data.smiles[0])
    # atom_weights here are used to show for which atom the explanation is given
    atom_weights = np.zeros(shape=len(left_weights))
    if n_atom is not None:
        atom_weights[n_atom] = 1

    # make figure
    if model_an is not None:
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # plot molecule
    axes[0][0].imshow(img_for_mol(mol, atom_weights=atom_weights))
    axes[0][0].set_title(data.smiles[0] if len(data.smiles[0])<=38 else data.smiles[0][:35] + '...')
    axes[0][0].set_axis_off()
    
    axes[0][1].set_title("Adjacency matrix")
    try:
        axes[0][1].imshow(to_scipy_sparse_matrix(data.edge_index).toarray().astype(float))
    except AttributeError:
        axes[0][1].imshow(scipy.sparse.csr_matrix(([1]*len(data.edge_index[0]), data.edge_index)).toarray())
        
    axes[0][2].set_title('Feature matrix')
    try:
        axes[0][2].imshow(data.x.cpu().detach().numpy())
    except AttributeError:
        axes[0][2].imshow(data.x)

    axes[1][0].set_title(titles.left)
    axes[1][0].imshow(img_for_mol(mol, atom_weights=left_weights))
    axes[1][0].set_axis_off()

    axes[1][1].set_title(titles.centre)
    axes[1][1].imshow(img_for_mol(mol, atom_weights=centre_weights))
    axes[1][1].set_axis_off()

    axes[1][2].set_title(titles.right)
    axes[1][2].imshow(img_for_mol(mol, atom_weights=right_weights))
    axes[1][2].set_axis_off()


    # dodatkowa analiza
    if model_an is not None:
        at_w = my_scale( np.array(explanations.sal_increase)-np.array(explanations.sal_decrease) )
        
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[2][0].set_title("Positive - negative saliency map")
        axes[2][0].imshow(img_for_mol(mol, atom_weights=at_w))
        axes[2][0].set_axis_off()
        
        axes[2][1].set_title("Partial derivatives total sum")
        axes[2][1].bar(np.array(range(model_an.input.grad.shape[1])), sum(model_an.input.grad).cpu())
        axes[2][2].set_title("Partial derivatives")
        im  = axes[2][2].imshow(model_an.input.grad.cpu())
        cb = plt.colorbar(im)
#         plt.show()
    
    plt.savefig(os.path.join(storage, 'explanations', f'{identifier}.pdf'), bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')

    
def img_for_mol(mol, atom_weights=[]):
    """plot mol with atom weights"""
    highlight_kwargs = {}
    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('bwr')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        }
        highlight_kwargs = {
            'highlightAtoms': list(range(len(atom_weights))),
            'highlightBonds': [],
            'highlightAtomColors': atom_colors
        }

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(750, 750)
    drawer.SetFontSize(1)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, **highlight_kwargs)
                        
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    tempname = f'tmp-{random.randint(1, 100000)}.png'
    svg2png(bytestring=svg, write_to=tempname, dpi=600)
    img = imread(tempname)
    os.remove(tempname)
    return img