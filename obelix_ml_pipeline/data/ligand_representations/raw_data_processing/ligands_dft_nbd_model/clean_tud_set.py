# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import pandas as pd
import numpy as np


def preprocess_tud_dft_descriptors(df):
    df["min_NBO_donor"] = df[["nbo_charge_max_donor_dft", "nbo_charge_min_donor_dft"]].min(axis=1)
    df["max_NBO_donor"] = df[["nbo_charge_max_donor_dft", "nbo_charge_min_donor_dft"]].max(axis=1)

    df["min_bv_donor"] = df[["buried_volume_donor_min", "buried_volume_donor_max"]].min(axis=1)
    df["max_bv_donor"] = df[["buried_volume_donor_min", "buried_volume_donor_max"]].max(axis=1)

    df["Rh_donor_min_d"] = df[["distance_Rh_max_donor_gaussian", "distance_Rh_min_donor_gaussian"]].min(axis=1)
    df["Rh_donor_max_d"] = df[["distance_Rh_min_donor_gaussian", "distance_Rh_max_donor_gaussian"]].max(axis=1)

    df['std_quad'] = df[["NE_quad", "NW_quad", "SW_quad", "SE_quad"]].std(axis=1)
    df['min_quad'] = df[["NE_quad", "NW_quad", "SW_quad", "SE_quad"]].min(axis=1)
    df['max_quad'] = df[["NE_quad", "NW_quad", "SW_quad", "SE_quad"]].max(axis=1)

    df['std_oct'] = df[
        ['+,+,+_octant', '-,+,+_octant', '-,-,+_octant', '+,-,+_octant', '+,-,-_octant', '-,-,-_octant', '-,+,-_octant',
         '+,+,-_octant']].std(axis=1)
    df['min_oct'] = df[
        ['+,+,+_octant', '-,+,+_octant', '-,-,+_octant', '+,-,+_octant', '+,-,-_octant', '-,-,-_octant', '-,+,-_octant',
         '+,+,-_octant']].min(axis=1)
    df['max_oct'] = df[
        ['+,+,+_octant', '-,+,+_octant', '-,-,+_octant', '+,-,+_octant', '+,-,-_octant', '-,-,-_octant', '-,+,-_octant',
         '+,+,-_octant']].max(axis=1)

    df['ratio_std_oct_std_quad'] = df['std_oct'] / df['std_quad']
    df['ratio_min_oct_min_quad'] = df['min_oct'] / df['min_quad']

    df["lone_pair_occ_min"] = df[["lone_pair_occupancy_max_donor_dft", "lone_pair_occupancy_min_donor_dft"]].min(axis=1)
    df["lone_pair_occ_max"] = df[["lone_pair_occupancy_max_donor_dft", "lone_pair_occupancy_min_donor_dft"]].max(axis=1)

    # create bite_angle_sin and bite_angle_cos columns which are between -1 and 1 instead of 0 to 360
    # take sinus directly on angle
    df['bite_angle_sin'] = np.sin(df['bite_angle'])
    df['bite_angle_cos'] = np.cos(df['bite_angle'])

    # take sinus on angle and divide by 2 and add 0.5 to get values between 0 and 1 (not used)
    # df['bite_angle_rad'] = np.radians(df['bite_angle'])
    # df['bite_angle_sin'] = np.sin(df['bite_angle_rad']) / 2 + 0.5
    # df['bite_angle_cos'] = np.cos(df['bite_angle_rad']) / 2 + 0.5

    # create cone_angle_sin and cone_angle_cos columns which are between 0 and 1 instead of 0 to 360
    df['cone_angle_sin'] = np.sin(df['cone_angle'])
    df['cone_angle_cos'] = np.cos(df['cone_angle'])

    # take sinus on angle and divide by 2 and add 0.5 to get values between 0 and 1 (not used)
    # df['cone_angle_rad'] = np.radians(df['cone_angle'])
    # df['cone_angle_sin'] = np.sin(df['cone_angle_rad']) / 2 + 0.5
    # df['cone_angle_cos'] = np.cos(df['cone_angle_rad']) / 2 + 0.5

    # calculate the absolute difference between free_ligand_nbo_charge_max_donor_dft and nbo_charge_max_donor_dft
    df['nbo_charge_max_donor_dft_abs_diff'] = abs(df['free_ligand_nbo_charge_max_donor_dft'] - df['nbo_charge_max_donor_dft'])
    # calculate the absolute difference between free_ligand_nbo_charge_min_donor_dft and nbo_charge_min_donor_dft
    df['nbo_charge_min_donor_dft_abs_diff'] = abs(df['free_ligand_nbo_charge_min_donor_dft'] - df['nbo_charge_min_donor_dft'])

    # calculate the absolute difference between free_ligand_mulliken_charge_max_donor_dft and mulliken_charge_max_donor_dft
    df['mulliken_charge_max_donor_dft_abs_diff'] = abs(df['free_ligand_mulliken_charge_max_donor_dft'] - df['mulliken_charge_max_donor_dft'])
    # calculate the absolute difference between free_ligand_mulliken_charge_min_donor_dft and mulliken_charge_min_donor_dft
    df['mulliken_charge_min_donor_dft_abs_diff'] = abs(df['free_ligand_mulliken_charge_min_donor_dft'] - df['mulliken_charge_min_donor_dft'])

    # calculate the absolute difference between free_ligand_lone_pair_occupancy_max_donor_dft and lone_pair_occupancy_max_donor_dft
    df['lone_pair_occupancy_max_donor_dft_abs_diff'] = abs(df['free_ligand_lone_pair_occupancy_max_donor_dft'] - df['lone_pair_occupancy_max_donor_dft'])
    # calculate the absolute difference between free_ligand_lone_pair_occupancy_min_donor_dft and lone_pair_occupancy_min_donor_dft
    df['lone_pair_occupancy_min_donor_dft_abs_diff'] = abs(df['free_ligand_lone_pair_occupancy_min_donor_dft'] - df['lone_pair_occupancy_min_donor_dft'])
    return df


def apply_preprocessing(df):
    df = df.loc[:, ~df.columns.str.contains('index|idx|element|time|optimization')]  # remove elements, indices, optimization info
    df = df.loc[:,~df.columns.str.contains('orbital_occupation')]  # remove descriptors that are prone to introducing errors
    # this is for clean_Rh_ligand_NBD_DFT_descriptors_v3.xlsx
    # df = df.drop(['Code', 'Ligand alias', 	'CAS', 	'Formula',	'Class',	'Eq to Rh', 'Canonical SMILES', 'Isomeric SMILES', 'InChI', 'InChI Key', 'InChI key main layer',
    #                                             'filename_tud', 'cas_or_ligand#', 'cas', 'optimization_success_dft'], axis=1)  # remove redundant columns

    # check if all values per column are finite (if column is numerical) and not NaN, else print the column name
    for col in df.columns:
        if df[col].dtype.kind in 'ifc' and np.isfinite(df[col]).all() and not np.isnan(df[col]).all():
            pass
        else:
            print(col)

    # write df to file
    # df.to_csv('../ligands_dft_nbd_model.csv', index=False)
    return df


def preprocess_free_ligand_descriptors(df):
    # keep Ligand# column and drop columns containing 'time', 'orbital_occupation', 'index', 'idx', 'element'
    df = df.loc[:, ~df.columns.str.contains('time|orbital_occupation|index|idx|element|optimization|filename')]
    # drop columns with thermodynamic descriptors such that we are only left with electronic descriptors
    df = df.loc[:, ~df.columns.str.contains('entropy|zero_point|sum_electronic')]
    # print all rows that contain NaN values
    print(df[df.isna().any(axis=1)])
    return df


if __name__ == "__main__":
    df = pd.read_csv('clean_Rh_ligand_NBD_DFT_descriptors_v8.csv')
    df = apply_preprocessing(df)
    free_ligand_df = pd.read_csv('free_ligand_descriptors_v3.csv')
    free_ligand_df = preprocess_free_ligand_descriptors(free_ligand_df)
    # merge ../ligands_dft_nbd_model.csv and free_ligand_descriptors_v1.csv on Ligand# column
    df = df.merge(free_ligand_df, on='Ligand#')
    df = preprocess_tud_dft_descriptors(df)  # process the descriptors to create additional representations of the data
    df.to_csv('../ligands_dft_nbd_model.csv', index=False)
