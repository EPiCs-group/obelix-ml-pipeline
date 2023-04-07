import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from morfeus import Sterimol, read_xyz
import numpy as np

def calc_sterimol_descr(name_list):
    """Calculate sterimol parameters from xyz files
    and create a descriptor dataframe with names as index
    :name_list: list of names
    :return: a dataframe with names & descriptors
    """
    col = ['L','B_1','B_5','L_uncorr','L_a','L_b','L_c','B_1_a','B_1_b','B_1_c','B_5_a','B_5_b','B_5_c']
    ster = []
    for sm in name_list:
        elements, coordinates = read_xyz(sm+'.xyz')
        sterimol = Sterimol(elements, coordinates, 1, 2) #, radii=None, radii_type='bondi', n_rot_vectors=3600, excluded_atoms=None, calculate=True)
        ster.append([sterimol.L_value]+ [sterimol.B_1_value]+[sterimol.B_5_value]+[sterimol.L_value_uncorrected]+list(sterimol.L)+list(sterimol.B_1)+list(sterimol.B_5))
    descr_df = pd.DataFrame(ster, index = name_list, columns = col)
    descr_df = descr_df.reset_index(drop=False)
    return descr_df

def calc_fps_descr(smiles_list, name_list):
    """Calculate RDKit Morgan Fingerprints from smiles list
    and create a descriptor dataframe with names as index
    :smiles_list: list of smiles
    :name_list: list of names
    :return: a dataframe with names & descriptors
    """
    dict_descr = dict()
    for idx,smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        name = name_list[idx]
        if mol:
            #dd = descr.calc_mol(mol)
            dd = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=1024))
            dict_descr[name] = dd
        else:
            dict_descr[name] = [None]*1024
    # make rdkit mols
    descr_df = pd.DataFrame(dict_descr).T
    # name columns to keep track of deleted bits
    descr_df.columns = [f'fp{n}' for n in range(1024)]
    descr_df = descr_df.reset_index(drop=False)
    # remove constant columns
    descr_df = descr_df.loc[:, (descr_df != descr_df.iloc[0]).any()] 
    return descr_df

def calc_rdkit_descr(smiles_list, name_list):
    """Calculate RDKit descriptors from smiles list
    and create a descriptor dataframe with names as index
    :smiles_list: list of smiles
    :name_list: list of names
    :return: a dataframe with names & descriptors
    """
    dict_descr = dict()
    for idx,smi in enumerate(smiles_list):
        descr = RDKitDescriptors()
        mol = Chem.MolFromSmiles(smi)
        name = name_list[idx]
        if mol:
            dd = descr.calc_mol(mol)
            dict_descr[name] = dd
        else:
            dict_descr[name] = [None]*len(descr.desc_names)
    # make rdkit mols
    descr_df = pd.DataFrame(dict_descr).T
    descr_df.columns = descr.desc_names
    descr_df = descr_df.reset_index(drop=False)
    # remove constant columns
    descr_df = descr_df.loc[:, (descr_df != descr_df.iloc[0]).any()] 
    return descr_df

# Code borrowed from
# https://github.com/PatWalters/useful_rdkit_utils
FUNCS = {name: func for name, func in Descriptors.descList}
def apply_func(name, mol):
    """Apply an RDKit descriptor calculation to a molecule
    :param name: descriptor name
    :param mol: RDKit molecule
    :return:
    """
    try:
        return FUNCS[name](mol)
    except:
        logging.exception("function application failed (%s->%s)", name, Chem.MolToSmiles(mol))
        return None

class RDKitDescriptors:
    """ Calculate RDKit descriptors"""

    def __init__(self):
        self.desc_names = [desc_name for desc_name, _ in sorted(Descriptors.descList)]

    def calc_mol(self, mol):
        """Calculate descriptors for an RDKit molecule
        :param mol: RDKit molecule
        :return: a numpy array with descriptors
        """
        res = [apply_func(name, mol) for name in self.desc_names]
        return np.array(res, dtype=float)

    def calc_smiles(self, smiles):
        """Calculate descriptors for a SMILES string
        :param smiles: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return self.calc_mol(mol)
        else:
            return None

df = pd.read_csv('substrates_smiles.csv')
df_ecfp = calc_fps_descr(df.SMILES.values, df.ID.values)
df_rdkit = calc_rdkit_descr(df.SMILES.values, df.ID.values)
df_sterimol = calc_sterimol_descr(df.ID.values)
# write df to file
df_ecfp.to_csv('../substrates_ecfp.csv', sep = ';', index=False)
df_rdkit.to_csv('../substrates_rdkit.csv', sep = ';', index=False)
df_sterimol.to_csv('../substrates_sterimol.csv', sep = ';', index=False)