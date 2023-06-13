# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

# available ligand and substrate representations
AVAILABLE_LIGAND_REPRESENTATION_TYPES = ['dft_nbd_model', 'dft_nbd_model_with_solvation', 'ecfp', 'ohe']
AVAILABLE_SUBSTRATE_REPRESENTATION_TYPES = ['dft_steric_fingerprint', 'smiles_steric_fingerprint', 'ecfp', 'ohe']

# selected features for ligand representations
# descriptors for DFT NBD model selected by two scientists
descriptor_selection_test_scientist_1 = ['buried_volume_donor_max', 'buried_volume_donor_min', 'nbo_charge_Rh_dft',
                                         'min_NBO_donor', 'max_NBO_donor', 'lone_pair_occ_max']

descriptor_selection_test_scientist_2 = []

# list of all DFT NBD model descriptors
all_steric_params = ['NE_quad', 'NW_quad', 'SW_quad',
                     'SE_quad', '+,+,+_octant', '-,+,+_octant', '-,-,+_octant',
                     '+,-,+_octant', '+,-,-_octant', '-,-,-_octant', '-,+,-_octant',
                     '+,+,-_octant', 'buried_volume_Rh_3.5A', 'buried_volume_donor_max',
                     'buried_volume_donor_min', 'buried_volume_Rh_4A', 'buried_volume_Rh_5A',
                     'buried_volume_Rh_6A', 'buried_volume_Rh_7A', 'min_bv_donor', 'max_bv_donor', 'std_quad',
                     'min_quad', 'max_quad', 'std_oct', 'min_oct', 'max_oct',
                     'ratio_std_oct_std_quad', 'ratio_min_oct_min_quad', ]
all_geometric_params = ['dihedral_angle_1', 'dihedral_angle_2', 'bite_angle', 'cone_angle',
                        'distance_Rh_max_donor_gaussian', 'distance_Rh_min_donor_gaussian', 'sasa_gfn2_xtb',
                        'Rh_donor_min_d', 'Rh_donor_max_d', 'bite_angle_sin', 'bite_angle_cos',
                        'cone_angle_sin', 'cone_angle_cos']
all_electronic_and_thermodynamic_params = ['distance_pi_bond_1', 'distance_pi_bond_2', 'dispersion_p_int_Rh_gfn2_xtb',
                                           'dispersion_p_int_donor_max_gfn2_xtb',
                                           'dispersion_p_int_donor_min_gfn2_xtb', 'ip_gfn2_xtb',
                                           'dipole_gfn2_xtb', 'ea_gfn2_xtb', 'electrofugality_gfn2_xtb',
                                           'nucleofugality_gfn2_xtb', 'nucleophilicity_gfn2_xtb',
                                           'electrophilicity_gfn2_xtb', 'HOMO_LUMO_gap_gfn2_xtb',
                                           'sum_electronic_and_free_energy_dft', 'sum_electronic_and_enthalpy_dft',
                                           'zero_point_correction_dft', 'entropy_dft', 'dipole_moment_dft',
                                           'lone_pair_occupancy_min_donor_dft',
                                           'lone_pair_occupancy_max_donor_dft', 'dispersion_energy_dft',
                                           'nbo_charge_Rh_dft', 'nbo_charge_min_donor_dft',
                                           'nbo_charge_max_donor_dft', 'mulliken_charge_Rh_dft',
                                           'mulliken_charge_min_donor_dft', 'mulliken_charge_max_donor_dft',
                                           'homo_energy_dft', 'lumo_energy_dft', 'homo_lumo_gap_dft',
                                           'hardness_dft', 'softness_dft', 'electronegativity_dft',
                                           'electrophilicity_dft', 'min_NBO_donor', 'max_NBO_donor',
                                           'lone_pair_occ_min',
                                           'lone_pair_occ_max']
all_descriptors = all_steric_params + all_geometric_params + all_electronic_and_thermodynamic_params

# selected DFT NBD model descriptors for ML based on correlations
steric_params_dft_nbd = ['NE_quad', 'NW_quad', 'SW_quad',
                         'SE_quad', '+,+,+_octant', '-,+,+_octant', '-,-,+_octant',
                         '+,-,+_octant', '+,-,-_octant', '-,-,-_octant', '-,+,-_octant',
                         '+,+,-_octant', 'buried_volume_Rh_3.5A', 'buried_volume_donor_max',
                         'buried_volume_donor_min']
geometric_params_dft_nbd = ['bite_angle_sin', 'bite_angle_cos', 'cone_angle_sin', 'cone_angle_cos',
                            'dihedral_angle_1', 'dihedral_angle_2', 'distance_Rh_max_donor_gaussian',
                            'distance_Rh_min_donor_gaussian']
electronic_and_thermodynamic_params_dft_nbd = ['distance_pi_bond_1', 'distance_pi_bond_2',
                                               'dispersion_p_int_Rh_gfn2_xtb', 'dispersion_p_int_donor_max_gfn2_xtb',
                                               'dispersion_p_int_donor_min_gfn2_xtb',
                                               'sum_electronic_and_free_energy_dft', 'dipole_moment_dft',
                                               'lone_pair_occupancy_min_donor_dft',
                                               'lone_pair_occupancy_max_donor_dft', 'dispersion_energy_dft',
                                               'nbo_charge_Rh_dft', 'nbo_charge_min_donor_dft',
                                               'nbo_charge_max_donor_dft', 'homo_lumo_gap_dft']
DFT_NBD_MODEL = steric_params_dft_nbd + geometric_params_dft_nbd + electronic_and_thermodynamic_params_dft_nbd

OLD_DFT_NBD_MODEL = ['bite_angle', 'cone_angle', 'buried_volume_Rh_4A', 'buried_volume_Rh_5A',
                     'nbo_charge_Rh_dft', 'buried_volume_Rh_6A', 'buried_volume_Rh_7A',
                     'dispersion_p_int_Rh_gfn2_xtb', 'dipole_moment_dft',
                     'dispersion_energy_dft', 'mulliken_charge_Rh_dft', 'homo_energy_dft', 'lumo_energy_dft',
                     'buried_volume_Rh_3.5A', 'dipole_gfn2_xtb',
                     'ea_gfn2_xtb', 'min_NBO_donor', 'max_NBO_donor', 'min_bv_donor', 'max_bv_donor',
                     'Rh_donor_min_d', 'Rh_donor_max_d',
                     'std_quad', 'min_quad', 'max_quad', 'std_oct', 'max_oct', 'min_oct', 'lone_pair_occ_max',
                     'lone_pair_occ_min', 'bite_angle_sin', 'bite_angle_cos',
                     'cone_angle_sin', 'cone_angle_cos', 'ratio_std_oct_std_quad', 'ratio_min_oct_min_quad']

# selected features for substrate representations
STERIMOL = ['L', 'B_1', 'B_5']
