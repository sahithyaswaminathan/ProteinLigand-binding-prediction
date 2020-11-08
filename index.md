# ProteinLigand Binding-prediction

```
Sahithya Swaminathan
2.12.2018
```
## Prerequisites

In order to run this script, install Python 2.7 and above

## Dataset

Dataset was extracted from Protein-Databank (PDB) and contains only X,Y,Z- coordinates of atom along with type of atom (hydrophobic, hydrophilic)

## Data-Preprocessing

Since each protein-ligand pair contains different number of atom size and type, it is important to pre-process the data to have a consistent atom number for administering into Convolution Neural Network model. A novel technique has been adopted to incorporate consistency. Min-Max normalization has been done on X,Y,Z-coordinates. A unit cube is divided into 20 bins and each atom with respective coordinates are accomodated in 20 bins. If the atom type in a specific bin has majority of of hydrophobic atoms, then the count of atom is encoded with -1 else, if hydrophilic atom count is greater than hydrophobic type, the count is encoded with +1.

## Convolution Neural Network

3-D CNN is built to train the model with Adam Optimizer and Cross-entropy loss function

The code was built using Keras with Tensorflow background
