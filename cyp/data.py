import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

def canonicalize(smiles):
    if '.' in smiles:
        smiles = sorted(smiles.split('.'), key=len)[-1]
    return Chem.CanonSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class CYPDataset(InMemoryDataset):
    def __init__(self, root, cyp='CYP3A4_1W0E', transform=None, pre_transform=None):
        self.cyp = cyp
        super(CYPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.cyp}.smi', f'{self.cyp}_node_values.npy']

    @property
    def processed_file_names(self):
        return [f'{self.cyp}_graphs.pt']

    # @debug_wrapper
    def download(self):
        src_path = os.path.join(self.root, f'{self.cyp}_clean.csv')
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"There is no file `{src_path}_clean.csv`")
        df = pd.read_csv(src_path)
        # default modifications:
        # ['Br', 'C', 'C(=O)C', 'C(=O)O', 'C(C)C', 'CC', 'CO', 'COC', 'Cl', 'F', 'I', 'N', 'O', 'OC', 'S']
        mods = sorted(df.modification.unique().tolist())
        n_mods = len(mods)

        smileses, ys = [], []
        errs = 0

        for smiles_p in tqdm(sorted(df.smiles_P.unique())):
            smiles_f = canonicalize(smiles_p)
            mol_p = Chem.MolFromSmiles(smiles_f)
            y = np.full((mol_p.GetNumAtoms(), n_mods), np.nan)
            df_filtered = df[df.smiles_P == smiles_p]
            error = False
            for i, row in df_filtered.iterrows():
                mol = Chem.MolFromSmiles(row.smiles)  # z podstawnikiem
                mapping = list(mol.GetSubstructMatch(mol_p))
                if not mapping:
                    error = True
                    errs += 1
                    break
                mod_idx = [i for i in range(mol.GetNumAtoms()) if i not in mapping]
                for bond in mol.GetBonds():
                    if bond.GetBeginAtomIdx() in mapping and bond.GetEndAtomIdx() in mod_idx:
                        sub_idx = bond.GetBeginAtomIdx()
                        break
                    elif bond.GetBeginAtomIdx() in mod_idx and bond.GetEndAtomIdx() in mapping:
                        sub_idx = bond.GetEndAtomIdx()
                        break
                else:
                    raise Error()
                y[mapping.index(sub_idx), mods.index(row.modification)] = row.docking_score - row.docking_score_P
            if not error:
                ys.append(y)
                smileses.append(smiles_f)
        with open(self.raw_paths[0], 'w') as file:
            file.write('\n'.join(smileses))
        np.save(self.raw_paths[1], ys)

    def process(self):
        data_list = []

        with open(self.raw_paths[0], 'r') as file:
            smileses = file.read().split('\n')
        ys = np.load(self.raw_paths[1], allow_pickle=True)
        for smiles, y in zip(smileses, ys):
            mol = Chem.MolFromSmiles(smiles)

            edges = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            edges = np.array(edges)

            nodes = []
            for atom in mol.GetAtoms():
                results = one_of_k_encoding_unk(
                    atom.GetSymbol(),
                    [
                        'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Unknown'
                    ]
                ) + one_of_k_encoding(
                    atom.GetDegree(),
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ) + one_of_k_encoding_unk(
                    atom.GetImplicitValence(),
                    [0, 1, 2, 3, 4, 5, 6]
                ) + [
                              atom.GetFormalCharge(),
                              atom.GetNumRadicalElectrons()
                          ] + one_of_k_encoding_unk(
                    atom.GetHybridization(),
                    [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2
                    ]
                ) + [
                              atom.GetIsAromatic()
                          ] + one_of_k_encoding_unk(
                    atom.GetTotalNumHs(),
                    [0, 1, 2, 3, 4]
                )
                nodes.append(results)
            nodes = np.array(nodes)

            datum = Data(
                x=torch.FloatTensor(nodes),
                edge_index=torch.LongTensor(edges).t(),
                y=torch.FloatTensor(y),
                smiles=smiles
            )

            data_list.append(datum)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
