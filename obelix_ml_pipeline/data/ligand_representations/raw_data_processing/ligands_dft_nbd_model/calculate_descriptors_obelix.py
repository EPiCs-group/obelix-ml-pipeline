# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import os
import glob
import pandas as pd
from tqdm import tqdm
from morfeus.io import read_cclib

from obelix.descriptor_calculator import Descriptors
from obelix.molecular_graph import molecular_graph
from obelix.free_ligand import FreeLigand
from obelix.tools.utilities import dataframe_from_dictionary

# calculate complex descriptors
path_to_dft_log_files = os.path.join(os.getcwd(), 'Workflow', 'Descriptors', 'DFT_gas_phase_jnj_naming')
dft_descriptors = Descriptors(central_atom='Rh', path_to_workflow=path_to_dft_log_files,
                              output_type='gaussian')
dft_descriptors.calculate_dft_descriptors_from_log(geom_type='BD', solvent=None, extract_xyz_from_log=True,
                                                   printout=False, metal_adduct='nbd', plot_steric_map=False)
dft_descriptors.descriptor_df.to_csv('test_commit_Rh_ligand_DFT_descriptors_free_ligand_prep.csv', index=False)

# extract free ligands from complex log files
# testing -- testing successful on xyz and log
complexes_to_calc_descriptors = glob.glob(os.path.join(path_to_dft_log_files, '*.log'))
dictionary_for_properties = {}

# iterate over log files and extract the free ligand as an xyz file
for metal_ligand_complex in tqdm(complexes_to_calc_descriptors):
    elements, coordinates = read_cclib(metal_ligand_complex)
    if not len(coordinates[-1]) == 3:  # if this is true, there is only 1 coordinates array
        coordinates = coordinates[-1]
    base_with_extension = os.path.basename(metal_ligand_complex)
    split_base = os.path.splitext(base_with_extension)
    filename = split_base[0]
    print(molecular_graph(elements=elements, coordinates=coordinates, extract_ligand=True, path_to_workflow=path_to_dft_log_files, filename=filename))

# free ligand descriptors
# make a list of L1 to L192

ligand_number_list = [f'L{i}' for i in range(1, 193)]
dictionary_for_properties = {}
# read excel file with min/max numbering from complex
complex_descriptors_df = pd.read_excel(
    r"obelix-ml-pipeline\obelix_ml_pipeline\data\ligand_representations\raw_data_processing\ligands_dft_nbd_model\clean_Rh_ligand_NBD_DFT_descriptors_v9.xlsx",
    sheet_name="Sheet1")

for ligand_number in ligand_number_list:
    # free ligands are in test_free_ligand_extraction/v2
    free_ligand = FreeLigand(os.path.join('test_free_ligand_extraction', 'v2'), f'free_ligand_{ligand_number}.xyz',
                             f'free_ligand_{ligand_number}_SP.log')
    # read the min/max donor indices from the excel file and match them with free_ligand.complex_xyz_bidentate_1_idx and free_ligand.complex_xyz_bidentate_2_idx
    # if complex_xyz_bidentate_1 is min donor, then free_ligand_xyz_bidentate_1 is min donor for our free ligand class
    # first check if index_donor_min from excel is equal to complex_xyz_bidentate_1_idx or complex_xyz_bidentate_2_idx
    complex_bidentate_min_donor_idx = \
    complex_descriptors_df.loc[complex_descriptors_df['Ligand#'] == ligand_number, 'index_donor_min'].values[0]
    if complex_bidentate_min_donor_idx == free_ligand.complex_xyz_bidentate_1_idx:
        # if this is true, then the free ligand bidentate 1 is the min donor
        free_ligand.min_donor_idx = free_ligand.free_ligand_xyz_bidentate_1_idx
        free_ligand.max_donor_idx = free_ligand.free_ligand_xyz_bidentate_2_idx
    else:
        # otherwise the free ligand bidentate 2 is the min donor
        free_ligand.min_donor_idx = free_ligand.free_ligand_xyz_bidentate_2_idx
        free_ligand.max_donor_idx = free_ligand.free_ligand_xyz_bidentate_1_idx

    free_ligand.initialize_dft_extractor(free_ligand.dft_log_file, None, free_ligand.min_donor_idx,
                                         free_ligand.max_donor_idx, metal_adduct='pristine')
    # free_ligand.assign_min_max_donor_dft()
    properties = free_ligand.calculate_dft_descriptors()
    dictionary_for_properties[ligand_number] = properties
new_descriptor_df = dataframe_from_dictionary(dictionary_for_properties)
# reset the index and name that column to 'Ligand#' for consistency with the complex descriptor df
new_descriptor_df = new_descriptor_df.reset_index().rename(columns={'index': 'Ligand#'})
new_descriptor_df.to_csv('free_ligand_descriptors_v1.csv', index=False)
