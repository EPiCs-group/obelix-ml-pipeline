# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import os
import pandas as pd


# load available ligand representations
def load_ligand_representations(representation_type, columns_of_representation_to_select=None):
    available_ligand_representation_types = ['dft_nbd_model', 'dl_chylon', 'ecfp', 'sigman_moo']
    if not representation_type in available_ligand_representation_types:
        raise ValueError(f'Representation type {representation_type} not available. Available types are: {available_ligand_representation_types}')
    path_to_ligand_representations = os.path.join(os.path.dirname(__file__), 'data', 'ligand_representations', f'ligands_{representation_type}.csv')
    ligand_df = pd.read_csv(path_to_ligand_representations)
    if columns_of_representation_to_select is not None:
        try:
            ligand_df = ligand_df[columns_of_representation_to_select]
        except KeyError:
            raise KeyError(f'Columns {columns_of_representation_to_select} not available in representation type {representation_type}')
    return ligand_df


# load available substrate representations
def load_substrate_representations(representation_type, columns_of_representation_to_select=None):
    available_substrate_representation_types = ['dft_steric_fingerprint', 'dl_chylon', 'ecfp', 'rdkit', 'sterimol']
    if not representation_type in available_substrate_representation_types:
        raise ValueError(f'Representation type {representation_type} not available. Available types are: {available_substrate_representation_types}')
    path_to_substrate_representations = os.path.join(os.path.dirname(__file__), 'data', 'substrate_representations', f'substrates_{representation_type}.csv')
    substrate_df = pd.read_csv(path_to_substrate_representations)
    if columns_of_representation_to_select is not None:
        try:
            substrate_df = substrate_df[columns_of_representation_to_select]
        except KeyError:
            raise KeyError(f'Columns {columns_of_representation_to_select} not available in representation type {representation_type}')
    return substrate_df


# load experimental data
def load_experimental_response():
    path_to_experimental_response = os.path.join(os.path.dirname(__file__), 'data', 'experimental_response', 'jnjdata_sm12378_MeOH_16h.csv')
    experimental_response_df = pd.read_csv(path_to_experimental_response)
    return experimental_response_df