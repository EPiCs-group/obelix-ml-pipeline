# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import pandas as pd
from obelix_ml_pipeline.data.ligand_representations.raw_data_processing.ligands_dft_nbd_model.clean_tud_set import apply_preprocessing

df = pd.read_excel('clean_Rh_ligand_NBD_DFT_descriptors_with_solvation.xlsx', sheet_name='Sheet1')
df = apply_preprocessing(df)