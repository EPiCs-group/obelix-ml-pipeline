import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from morfeus import Sterimol, read_xyz
# from rdkit import RDLogger
import numpy as np
# RDLogger.DisableLog('rdApp.*')
# RDLogger.EnableLog('rdApp.*')


def calc_fps_descr_from_sdf(sdf_filename_list, name_list):
    """Calculate RDKit Morgan Fingerprints from smiles list
    and create a descriptor dataframe with names as index
    :smiles_list: list of smiles
    :name_list: list of names
    :return: a dataframe with names & descriptors
    """
    dict_descr = dict()
    for idx,smi in enumerate(sdf_filename_list):
        try:
            sdf_supplier = Chem.SDMolSupplier(f'nbd_dft_opt_structures/{smi}_NBD_DFT.sdf')
            # each sdf file contains only one molecule
            mol = sdf_supplier[0]
            name = name_list[idx]
            if mol:
                #dd = descr.calc_mol(mol)
                dd = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=1024))
                dict_descr[name] = dd
            else:
                print(f'Could not create mol from {smi}')
        except Exception as e:
            print(f'Could not read {smi}')
            print(e)
            continue
    # make rdkit mols
    descr_df = pd.DataFrame(dict_descr).T
    # name columns to keep track of deleted bits
    descr_df.columns = [f'fp{n}' for n in range(1024)]
    descr_df = descr_df.reset_index(drop=False)
    descr_df.rename(columns={'index':'Ligand#'}, inplace=True)
    # remove constant columns
    descr_df = descr_df.loc[:, (descr_df != descr_df.iloc[0]).any()]
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
    # make rdkit mols
    descr_df = pd.DataFrame(dict_descr).T
    # name columns to keep track of deleted bits
    descr_df.columns = [f'fp{n}' for n in range(1024)]
    descr_df = descr_df.reset_index(drop=False)
    descr_df.rename(columns={'index':'Ligand#'}, inplace=True)
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
    # make rdkit mols
    descr_df = pd.DataFrame(dict_descr).T
    descr_df.columns = descr.desc_names
    descr_df = descr_df.reset_index(drop=False)
    descr_df.rename(columns={'index':'Ligand#'}, inplace=True)
    # remove constant columns
    descr_df = descr_df.loc[:, (descr_df != descr_df.iloc[0]).any()] 
    return descr_df

def calc_sigman_descr(name_list):
    """Extract SigmanGroup's descriptors from file with
    overlapping structures (after manual enantiomer swapping)
    from:
    https://github.com/SigmanGroup/Multiobjective_Optimization
    :name_list: list of names
    :return: a dataframe with names & descriptors
    """
    sig_c2v_df = pd.read_excel(f'SigmanGroup_descriptors.xlsx', 'C2v')
    sig_symm_df = pd.read_excel(f'SigmanGroup_descriptors.xlsx', 'Symmetry adapted')
    sig_c2v_df.columns = list(sig_c2v_df.columns[:7]) + [i+'_c2v' for i in sig_c2v_df.columns[7:]]
    sig_symm_df.columns = list(sig_symm_df.columns[:7]) + [i+'_symm' for i in sig_symm_df.columns[7:]]
    df_sigman = pd.concat([sig_c2v_df, sig_symm_df.iloc[:,7:]], axis = 1)
    df_sigman = df_sigman.dropna(axis = 0)
    # remove constant columns
    df_sigman = df_sigman.loc[:, (df_sigman != df_sigman.iloc[0]).any()]
    df_sigman = df_sigman.drop(['Ligand alias', 'CAS','Formula','Sigman Ligand ID','Sigman ligand name','Class'], axis=1)
    
    df_sigman = df_sigman[df_sigman['Ligand#'].isin(name_list)]
    return df_sigman

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


if __name__ == "__main__":
    df = pd.read_csv('ligands_smiles.csv')
    df_ecfp = calc_fps_descr(df.SMILES.values, df.ID.values)
    df_rdkit = calc_rdkit_descr(df.SMILES.values, df.ID.values)
    df_sigman = calc_sigman_descr(df.ID.values)
    # write df to file
    df_ecfp.to_csv('../ligands_ecfp.csv', sep = ';', index=False)
    df_rdkit.to_csv('../ligands_rdkit.csv', sep = ';', index=False)
    df_sigman.to_csv('../ligands_sigmangroup.csv', sep = ';', index=False)
    # calculate ecfp from sdf files
    # df = pd.read_excel('ligands_dft_nbd_model/clean_Rh_ligand_NBD_DFT_descriptors_v3.xlsx', 'Sheet1')
    # df_ecfp = calc_fps_descr_from_sdf(df['filename_tud'].values, df['Ligand#'].values)
