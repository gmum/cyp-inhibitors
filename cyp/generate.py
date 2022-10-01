import argparse
from typing import Optional, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdChemReactions
from tqdm import tqdm


class Molecule:
    def __init__(self, structure: Union[str, Mol], metadata: dict = None,
                 precursor: Optional['Molecule'] = None):
        if isinstance(structure, str):
            self.smiles = structure
            self._mol = None
        elif isinstance(structure, Mol):
            self.mol = structure
        else:
            raise TypeError(f'Unrecognized structure type: `{type(structure).__name__}`.')
        self.metadata = metadata
        self.precursor = precursor

    @property
    def mol(self):
        if not self._mol:
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol

    @mol.setter
    def mol(self, value):
        self._mol = value
        self.smiles = Chem.MolToSmiles(self._mol)

    def __repr__(self):
        return self.smiles


def run_reaction(molecule, smirks):
    rxn = rdChemReactions.ReactionFromSmarts(smirks)
    reacts = (Chem.MolFromSmiles(molecule.smiles),)
    products = rxn.RunReactants(reacts)
    return [Molecule(p[0], precursor=molecule) for p in products]


class CombinatorialGenerator:
    def __init__(self):
        self.substituents = [
            'F',
            'Cl',
            'Br',
            'I',
            'C',
            'C(C)C',
            'CC',
            'C(=O)O',
            'O',
            'OC',
            'COC',
            'CO',
            'C(=O)C',
            'N',
            'S',
        ]

        self.reactions = [f'[*:3]:[ch:1]:[*:2]>>[*:3]:[c:1]({s}):[*:2]' for s in self.substituents] + [
            f'[*:3]~[Ch;R;D2:1]~[*:2]>>[*:3]~[C:1]({s})~[*:2]' for s in self.substituents]

    def generate(self, library):
        lib = []
        for molecule in tqdm(library):
            for reaction in self.reactions:
                products = run_reaction(molecule, reaction)
                lib.extend([p for p in products if p.smiles not in library])

        lib = list(set(lib))
        return lib


class RingSizeFilter:
    def __init__(self, max_ring_size):
        self.max_ring_size = max_ring_size

    def __call__(self, molecule):
        mol = molecule.mol
        mol.UpdatePropertyCache()
        Chem.FastFindRings(mol)
        return all(len(ring) <= self.max_ring_size for ring in mol.GetRingInfo().AtomRings())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    lib = pd.read_csv(args.input)
    if 'smiles' not in lib.columns:
        raise ValueError('The input file should contain a `smiles` column.')
    lib.drop_duplicates(subset='smiles', inplace=True)
    ring_size_filter = RingSizeFilter(8)
    mols = [Molecule(smiles) for smiles in lib.smiles]
    mols = [mol for mol in mols if ring_size_filter(mol)]
    gen = CombinatorialGenerator()
    mols = gen.generate(mols)
    pd.DataFrame(data={'smiles': [m.smiles for m in mols]}).to_csv(args.output)
