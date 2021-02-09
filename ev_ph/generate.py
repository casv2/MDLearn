from ase.io import read, write
import ase
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np
import phonopy

from quippy.potential import Potential
from ase.io import read, write
import sys

def get_phonon_configs(calc, at, disps, scell, config_type):

    cell = PhonopyAtoms(symbols=at.get_chemical_symbols(),
                    cell=at.get_cell(),
                    positions=at.get_positions())

    phonon = Phonopy(cell, np.eye(3)*scell)

    al = []

    for disp in disps:
        print(disp, config_type)
        phonon.generate_displacements(distance=disp)
        supercells = phonon.get_supercells_with_displacements()

        for (i,scell) in enumerate(supercells):
            at = ase.Atoms(symbols=scell.get_chemical_symbols(),
                      scaled_positions=scell.get_scaled_positions(),
                      cell=scell.get_cell(),
                      pbc=True)

            at.set_calculator(calc)
            e = at.get_potential_energy()
            f = at.get_forces()
            v = -1.0 * at.get_volume() * at.get_stress(voigt=False)
            at.arrays["force"] = f
            at.info["virial"] = v
            at.info["config_type"] = "PH_" + config_type
            at.info["energy_TB"] = e
            #write("PH_{}_{}_scell_{}.xyz".format(config_type, i, disp), at)
            al.append(at)

    return al

def get_FLD_configs(calc, in_at, config_type, Vs=np.linspace(0.96, 1.03, 6), Ds=np.linspace(-0.2, 0.2, 6)):
    at_in = read(in_at)

    al = []
    for i in Vs:
        for j in Ds:
            at = ase.Atoms(at_in.get_chemical_symbols(), positions = at_in.get_positions(), cell=at_in.get_cell(), pbc=[True,True,True])
            k, l = np.random.randint(0,3), np.random.randint(0,3)
            at.set_cell(at.cell * i, scale_atoms=True)
            at.positions = at.positions + np.random.normal(0, 0.1, (len(at),3))
            at.cell[k][l] += (j + np.random.normal(0, 0.2))
            at.set_calculator(calc)
            e = at.get_potential_energy()
            f = at.get_forces()
            #v = -1.0 * at.get_volume() * at.get_stress(voigt=False)
            print(i,j,config_type)
            print(e/len(at))
            print(at.get_volume()/len(at))
            at.arrays["force"] = f
            #at.info["virial"] = v
            at.info["config_type"] = "FLD_" + config_type
            at.info["energy_TB"] = e
            #write("hcp_FLD_{}_{}_{}_{}.xyz".format(i,j,k,l), at)
            al.append(at)
    
    return al

calc = Potential("TB NRL-TB", param_filename="./16x16x16k_mesh_param_file_tightbind.parms.NRL_TB.Ti_spline.xml")

# at_bcc = read("./NRLTB_bcc.xyz")
# at_hcp = read("./NRLTB_hcp.xyz")

# PH_bcc = get_phonon_configs(calc, at_bcc, [0.1], 3, "bcc")
# PH_hcp = get_phonon_configs(calc, at_hcp, [0.1], 3, "hcp")

FLD_bcc = get_FLD_configs(calc, "./NRLTB_bcc.xyz", "bcc")
FLD_hcp = get_FLD_configs(calc, "./NRLTB_hcp.xyz", "hcp")

al = FLD_bcc + FLD_hcp #+ PH_bcc + PH_hcp

write("./FLD16_hcp_bcc_sm3-6+3p.xyz", al)


# al = PH_bcc + PH_hcp

#write("./PH_bcc_hcp_disps.xyz", al)

# calc = Potential("TB NRL-TB", param_filename="./16x16x16k_mesh_param_file_tightbind.parms.NRL_TB.Ti_spline.xml")


# al = FLD_bcc + FLD_hcp + PH_bcc + PH_hcp

# write("./Ti_PH_FLD_bcc_hcp_TB.xyz", al)
