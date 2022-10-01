# Generation of new inhibitors of selected cytochrome P450 subtypes – in silico study

## Data

The poses of docked compounds are available in the `data/poses` directory. For each generated compound, its best scoring pose is saved in the mol2 data format.

In the `data/crystals` directory, there is a set of pre-processed pdb files containing CYP structures used for docking. The binding pocket centers are described in the corresponding yaml files.

## Visualizer

Our ligand-protein complex visualizer is available [here](https://gmum.github.io/cyp-inhibitors/)!

## Dependencies

- numpy
- pandas
- rdkit
- scikit-learn
- pytorch
- pytorch-geometric
- tqdm
- yaml

## Code usage

To run the experiments with the default
hyperparameters, use this command:

```bash
python -m cyp.train --cyp [CYPNAME]
```

where CYPNAME is any of these CYP crystals:
- CYP2C8_2NNI
- CYP2C8_2PQ2
- CYP2C9_1OG2
- CYP2C9_4NZ2
- CYP2D6_2F9Q
- CYP2D6_3QM4
- CYP3A4_1W0E
- CYP3A4_1W0G

The hyperparameters can be changed by passing a grid search
file to the `--grid-file` argument. They can also
be input directly as command line arguments. For the
list of arguments, run:

```bash
python -m cyp.train --help
```

## Derivatives Generation

To generate molecule derivatives from your own file,
prepare a CSV file containing a `smiles` column (it
needs to be named exactly `smiles`), and run this script:

```bash
python -m cyp.generate --input [YOUR_CSV_FILE_PATH] --output [OUTPUT_CSV_FILE_PATH]
```