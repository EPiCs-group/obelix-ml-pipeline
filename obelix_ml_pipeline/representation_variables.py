# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# available ligand and substrate representations
AVAILABLE_LIGAND_REPRESENTATION_TYPES = ['dft_nbd_model', 'dft_nbd_model_with_solvation', 'dl_chylon', 'ecfp', 'sigmangroup','ohe', 'dft_nbd_model_fairsubset', 'dft_nbd_model_with_solvation_fairsubset', 'dl_chylon_fairsubset', 'ecfp_fairsubset', 'sigmangroup_fairsubset','ohe_fairsubset',]
AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES = ['dft_steric_fingerprint', 'smiles_steric_fingerprint', 'dl_chylon', 'ecfp', 'rdkit', 'sterimol', 'ohe']

# selected features for ligand representations
DFT_NBD_MODEL = ['bite_angle', 'cone_angle', 'buried_volume_Rh_4A', 'buried_volume_Rh_5A',
                   'nbo_charge_Rh_dft', 'buried_volume_Rh_6A', 'buried_volume_Rh_7A',
                   'dispersion_p_int_Rh_gfn2_xtb', 'dipole_moment_dft',
                   'dispersion_energy_dft', 'mulliken_charge_Rh_dft', 'homo_energy_dft', 'lumo_energy_dft',
                   'buried_volume_Rh_3.5A', 'dipole_gfn2_xtb',
                   'ea_gfn2_xtb', 'min_NBO_donor', 'max_NBO_donor', 'min_bv_donor', 'max_bv_donor',
                   'Rh_donor_min_d', 'Rh_donor_max_d',
                   'std_quad', 'min_quad', 'max_quad', 'std_oct', 'max_oct', 'min_oct', 'lone_pair_occ_max',
                   'lone_pair_occ_min', 'bite_angle_sin', 'bite_angle_cos',
                   'cone_angle_sin', 'cone_angle_cos', 'ratio_std_oct_std_quad', 'ratio_min_oct_min_quad']

# ToDo: add selection for sigmangroup descriptors

# selected features for substrate representations
STERIMOL = ['L', 'B_1', 'B_5']

