# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import os
import pandas as pd
from obelix_ml_pipeline.representation_variables import AVAILABLE_LIGAND_REPRESENTATION_TYPES, AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES
from obelix_ml_pipeline.utilities import load_csv_or_excel_file_to_df, merge_dfs, plot_dendrogram_for_substrate_rep
from obelix_ml_pipeline.representation_variables import AVAILABLE_LIGAND_REPRESENTATION_TYPES, AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES, DFT_NBD_MODEL, STERIMOL


def load_and_merge_representations_and_experimental_response(selected_ligand_representations, selected_substrate_representations, plot_dendrograms=False):
    # first load ligand representation and experimental response, if multiple representations are selected, loop over
    # them and keep merging the dataframes. Afterward load substrate representation and if multiple are selected do
    # the same. For each ligand representation also plot a dendrogram of the representation or combination of representations

    # load ligand representation
    # features to select from the ligand representation, these are defined in constants.py per representation type
    selected_features = select_features_for_representation(selected_ligand_representations[0], ligand=True)
    ligand_index_column = 'Ligand#'
    selected_features_and_ligand_index = selected_features + [ligand_index_column]
    first_ligand_rep = load_ligand_representations(selected_ligand_representations[0], columns_of_representation_to_select=selected_features_and_ligand_index)
    exp_df = load_experimental_response()
    ligand_rep_and_exp_df = merge_dfs(exp_df, ligand_index_column, first_ligand_rep, ligand_index_column)
    if len(selected_ligand_representations) > 1:
        for selected_ligand_representation in selected_ligand_representations[1:]:
            selected_features = select_features_for_representation(selected_ligand_representation, ligand=True)
            selected_features_and_ligand_index = selected_features + [ligand_index_column]
            ligand_rep = load_ligand_representations(selected_ligand_representation, columns_of_representation_to_select=selected_features_and_ligand_index)
            # keep merging ligand rep to first ligand rep and exp df
            ligand_rep_and_exp_df = merge_dfs(ligand_rep_and_exp_df, ligand_index_column, ligand_rep, ligand_index_column)

    # load substrate representation
    # features to select from the substrate representation, these are defined in constants.py per representation type
    selected_features = select_features_for_representation(selected_substrate_representations[0], ligand=False)
    substrate_index_column_substrate_df = 'index'
    selected_features_and_substrate_index = selected_features + [substrate_index_column_substrate_df]
    first_substrate_rep = load_substrate_representations(selected_substrate_representations[0], columns_of_representation_to_select=selected_features_and_substrate_index)
    # check similarity of substrates based on first selected substrate representation
    if plot_dendrograms:
        plot_dendrogram_for_substrate_rep(first_substrate_rep, selected_substrate_representations[0])
    substrate_index_column_exp_df = 'Substrate'
    substrate_rep_ligand_rep_and_exp_df = merge_dfs(ligand_rep_and_exp_df, substrate_index_column_exp_df, first_substrate_rep, substrate_index_column_substrate_df)
    substrate_rep = first_substrate_rep
    # do the same for the other selected substrate representations
    if len(selected_substrate_representations) > 1:
        for selected_substrate_representation in selected_substrate_representations[1:]:
            selected_features = select_features_for_representation(selected_substrate_representation, ligand=False)
            selected_features_and_substrate_index = selected_features + [substrate_index_column_substrate_df]
            next_substrate_rep = load_substrate_representations(selected_substrate_representation, columns_of_representation_to_select=selected_features_and_substrate_index)
            # plot dendrogram for this substrate representation
            if plot_dendrograms:
                plot_dendrogram_for_substrate_rep(next_substrate_rep, selected_substrate_representation)
            # merge substrate rep to substrate rep to plot dendrogram of all substrate reps combined
            substrate_rep = merge_dfs(substrate_rep, substrate_index_column_substrate_df, next_substrate_rep, substrate_index_column_substrate_df)
            # keep merging substrate rep to first substrate rep and ligand rep and exp df
            substrate_rep_ligand_rep_and_exp_df = merge_dfs(substrate_rep_ligand_rep_and_exp_df, substrate_index_column_substrate_df, next_substrate_rep, substrate_index_column_substrate_df)

    # drop 'index' column since 'Substrate' column from experimental response contains substrate names
    substrate_rep_ligand_rep_and_exp_df = substrate_rep_ligand_rep_and_exp_df.drop('index', axis=1)

    # plot dendrogram for all substrate representations combined
    if plot_dendrograms:
        if len(selected_substrate_representations) > 1:
            combination_of_substrate_rep_types = ' + '.join(selected_substrate_representations)
            plot_dendrogram_for_substrate_rep(substrate_rep, combination_of_substrate_rep_types)

    # final Nan check, dropping can be done outside this function since not all columns might be used for ML
    # print Ligand# of rows with Nan values and the columns that are Nan for each row
    # print(f'Ligand# of rows with Nan values: {substrate_rep_ligand_rep_and_exp_df[substrate_rep_ligand_rep_and_exp_df.isnull().any(axis=1)]["Ligand#"]}')
    # print(f'Columns with Nan values: {substrate_rep_ligand_rep_and_exp_df[substrate_rep_ligand_rep_and_exp_df.isnull().any(axis=1)]}')
    return substrate_rep_ligand_rep_and_exp_df


# load available ligand representations
def load_ligand_representations(representation_type, columns_of_representation_to_select=None):
    available_ligand_representation_types = AVAILABLE_LIGAND_REPRESENTATION_TYPES
    if not representation_type in available_ligand_representation_types:
        raise ValueError(f'Representation type {representation_type} not available. Available types are: {available_ligand_representation_types}')
    path_to_ligand_representations = os.path.join(os.path.dirname(__file__), 'data', 'ligand_representations', f'ligands_{representation_type}.csv')
    ligand_df = load_csv_or_excel_file_to_df(path_to_ligand_representations)
    if columns_of_representation_to_select is not None:
        try:
            ligand_df = ligand_df[columns_of_representation_to_select]
        except KeyError:
            raise KeyError(f'Columns {columns_of_representation_to_select} not available in representation type {representation_type}')
    return ligand_df


# load available substrate representations
def load_substrate_representations(representation_type, columns_of_representation_to_select=None):
    available_substrate_representation_types = AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES
    if not representation_type in available_substrate_representation_types:
        raise ValueError(f'Representation type {representation_type} not available. Available types are: {available_substrate_representation_types}')
    path_to_substrate_representations = os.path.join(os.path.dirname(__file__), 'data', 'substrate_representations', f'substrates_{representation_type}.csv')
    substrate_df = load_csv_or_excel_file_to_df(path_to_substrate_representations)
    if columns_of_representation_to_select is not None:
        try:
            substrate_df = substrate_df[columns_of_representation_to_select]
        except KeyError:
            raise KeyError(f'Columns {columns_of_representation_to_select} not available in representation type {representation_type}')
    return substrate_df


# load experimental data
def load_experimental_response():
    path_to_experimental_response = os.path.join(os.path.dirname(__file__), 'data', 'experimental_response', 'jnjdata_sm12378_MeOH_16h.csv')
    experimental_response_df = load_csv_or_excel_file_to_df(path_to_experimental_response)
    return experimental_response_df


def load_representation_and_return_all_columns_except_index(load_function, representation_type, columns_of_representation_to_select=None):
    df = load_function(representation_type, columns_of_representation_to_select=columns_of_representation_to_select)
    # return all columns except first one (which is the Ligand number)
    return df.columns[1:].tolist()


def select_features_for_representation(representation_type, ligand: bool):
    if representation_type in AVAILABLE_LIGAND_REPRESENTATION_TYPES and ligand:
        # these representations are loaded from representation_variables.py
        if representation_type == 'dft_nbd_model':
            return DFT_NBD_MODEL
        # these representations are always the same, so automatically determined
        if representation_type in ['ecfp', 'dl_chylon']:
            return load_representation_and_return_all_columns_except_index(load_ligand_representations, representation_type)
        return None
    elif representation_type in AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES and not ligand:
        # these representations are loaded from representation_variables.py
        if representation_type == 'sterimol':
            return STERIMOL
        # these representations are always the same, so automatically determined
        if representation_type in ['smiles_steric_fingerprint', 'dft_steric_fingerprint', 'dl_chylon', 'ecfp', 'rdkit']:
            return load_representation_and_return_all_columns_except_index(load_substrate_representations, representation_type)
        return None
    else:
        raise ValueError(f'Representation type {representation_type} not available. Available types are: '
                         f'{AVAILABLE_LIGAND_REPRESENTATION_TYPES} for ligands '
                         f'and {AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES} for substrates')


if __name__ == "__main__":
    # test loading of data
    selected_ligand_representations = ['dft_nbd_model']
    selected_substrate_representations = ['sterimol']
    df = load_and_merge_representations_and_experimental_response(selected_ligand_representations, selected_substrate_representations)
    print(df.groupby('Substrate').count())