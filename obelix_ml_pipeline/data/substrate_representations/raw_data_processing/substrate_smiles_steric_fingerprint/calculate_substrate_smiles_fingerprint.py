# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import pandas as pd
from morfeus.io import read_xyz
from morfeus import BuriedVolume, Sterimol


def calculate_steric_fingerprint(df):
    # list of possible substrates
    possible_substrates = df['Substrate'].unique()
    index_columns = [col for col in df.columns if 'index' in col]
    # iterate over the substrates to get filename from filename_tud, load xyz file and calculate buried volume at each R and C index
    for substrate in possible_substrates:
        print(substrate)
        # get the filename from the filename_tud column
        filename = df.loc[df['Substrate'] == substrate, 'filename_tud'].iloc[0]
        # load the xyz file
        elements, coordinates = read_xyz(filename)
        # each substrate has multiple R and C indices for example R1_index, R2_index, iterate over them and calculate the buried volume for each
        for index_column in index_columns:
            # get the index from the column
            index = df.loc[df['Substrate'] == substrate, index_column].iloc[0]
            # calculate the buried volume of 2A around the index
            buried_volume = BuriedVolume(elements, coordinates, index, radius=2).fraction_buried_volume
            # split this index on _ and append _bv to first part and write buried volume to this column
            df.loc[df['Substrate'] == substrate, index_column.split('_')[0] + '_bv'] = buried_volume

        # do the same for sterimol descriptors, but here we need C indices and R indices
        # remember that C1 is bound to R1 and R2, while C2 is bound to R3 and R4
        c_index_columns = [col for col in index_columns if 'C' in col]
        r_index_columns = [col for col in index_columns if 'R' in col]
        for c_index_column in c_index_columns:
            # get the index from the column
            c_index = df.loc[df['Substrate'] == substrate, c_index_column].iloc[0]
            # for this c_index iterate over r_index_columns to calculate sterimol over the pairs
            if c_index_column == 'C1_index':
                r_index_columns = ['R1_index', 'R2_index']
            elif c_index_column == 'C2_index':
                r_index_columns = ['R3_index', 'R4_index']
            for r_index_column in r_index_columns:
                # get the index from the column
                r_index = df.loc[df['Substrate'] == substrate, r_index_column].iloc[0]
                # calculate the sterimol descriptor between these two indices
                # print(c_index, r_index)
                sterimol = Sterimol(elements, coordinates, c_index, r_index)
                # print(f'sterimol values are {sterimol.L_value}, {sterimol.B_1_value}, {sterimol.B_5_value}')
                # # split this index on _ and append _L to first part and write sterimol L_value to this column
                df.loc[df['Substrate'] == substrate, c_index_column.split('_')[0] + '_' + r_index_column.split('_')[0] + '_L'] = sterimol.L_value
                # do the same for B_1_value and B_5_value
                df.loc[df['Substrate'] == substrate, c_index_column.split('_')[0] + '_' + r_index_column.split('_')[0] + '_B_1'] = sterimol.B_1_value
                df.loc[df['Substrate'] == substrate, c_index_column.split('_')[0] + '_' + r_index_column.split('_')[0] + '_B_5'] = sterimol.B_5_value

    # drop the index columns for the final dataframe
    df = df.drop(columns=index_columns + ['substrate_smiles', 'filename_tud'], axis=1)
    # rename 'Substrate' column to 'index' to match the other fingerprint files
    df = df.rename(columns={'Substrate': 'index'})
    # write the dataframe to a csv file
    df.to_csv('../substrates_smiles_steric_fingerprint.csv', index=False)


if __name__ == '__main__':
    # import pandas as pd
    # from rdkit import Chem
    # from rdkit.Chem import AllChem
    #
    # # Load the substrate data from the Excel file
    # df = pd.read_excel('substrate_atom_mapping.xlsx', sheet_name='Sheet1')
    #
    # # Loop through each row of the dataframe and generate an .xyz file for each substrate
    # for index, row in df.iterrows():
    #     # Get the SMILES string from the substrate_smiles column
    #     smiles = row['substrate_smiles']
    #
    #     # Use rdkit to convert the SMILES string to a molecule object
    #     mol = Chem.MolFromSmiles(smiles)
    #     mol = Chem.AddHs(mol)
    #
    #     # Use rdkit to generate the 3D coordinates of the molecule
    #     AllChem.EmbedMolecule(mol)
    #
    #     # Get the filename from the Substrate column
    #     filename = row['Substrate'] + '.xyz'
    #
    #     # Write the coordinates to an .xyz file with the same name as the substrate
    #     with open(filename, 'w') as f:
    #         f.write('{}\n\n'.format(mol.GetNumAtoms()))
    #         for atom in mol.GetAtoms():
    #             pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
    #             f.write('{} {} {} {}\n'.format(atom.GetSymbol(), pos.x, pos.y, pos.z))

    # read dataframe with substrate and indices
    df = pd.read_excel('substrate_atom_mapping.xlsx', sheet_name='Sheet1')
    # calculate steric fingerprint
    calculate_steric_fingerprint(df)


