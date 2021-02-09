from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np
import os
import sys
from ase.io import read, write
import pickle
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import pyjulip
import sys

config_type = sys.argv[1]
model_name = sys.argv[2]

#from quippy.potential import Potential
#calc = Potential("TB NRL-TB", param_filename="./quip_params.xml")

#os.path.join("..", model_name)

calc = pyjulip.ACE(model_name)

def get_crystal(config_type):
    if config_type == "bcc":
        at = read(os.path.join(os.path.abspath(os.path.dirname(__file__)), "bcc_sm.xyz"))
    else:
        at = read(os.path.join(os.path.abspath(os.path.dirname(__file__)), "hcp_sm.xyz"))
    #at = read("/Users/Cas/gits/ship-testing-framework/tests/Ti/phonon_Ti_bcc/bulk.xyz")

    cell = PhonopyAtoms(symbols=at.get_chemical_symbols(),
                    cell=at.get_cell(),
                    positions=at.get_positions())

    return cell

def phonopy_pre_process(cell, config_type, supercell_matrix=None):
    
    smat = [[3,0,0], [0,3,0], [0,0,3]]

    phonon = Phonopy(cell,
                     smat)

    phonon.generate_displacements(distance=0.16) #0.02
    print("[Phonopy] Atomic displacements:")
    disps = phonon.get_displacements()
    for d in disps:
        print("[Phonopy] %d %s" % (d[0], d[1:]))
    return phonon

def run(calc, phonon, config_type):
    supercells = phonon.get_supercells_with_displacements()
    # Force calculations by calculator
    set_of_forces = []
    for (i,scell) in enumerate(supercells):
        ##########
        at = Atoms(symbols=scell.get_chemical_symbols(),
                      scaled_positions=scell.get_scaled_positions(),
                      cell=scell.get_cell(),
                      pbc=True)

        at.set_calculator(calc)

        energy = at.get_potential_energy(force_consistent=True)
        forces = at.get_forces()
        stress = at.get_stress(voigt=False)

        drift_force = forces.sum(axis=0)
        print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
        # Simple translational invariance
        for force in forces:
            force -= drift_force / forces.shape[0]

        at.arrays["force"] = forces
        at.info["virial"] = -1.0 * at.get_volume() * stress
        at.info["energy"] = energy

        write("./Ti_{}_scell2.xyz".format(config_type), at)
        #write("{}_scell.xyz".format(i), at)
        set_of_forces.append(forces)
    return set_of_forces

cell = get_crystal(config_type)

print(config_type)

phonon = phonopy_pre_process(cell, config_type, supercell_matrix=np.eye(3, dtype='intc'))
set_of_forces = run(calc, phonon, config_type)
phonon.produce_force_constants(forces=set_of_forces)

path = [[[0, 0, 0], [-0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0.5, 0]]]
labels = ["$\\Gamma$", "H", "P", "$\\Gamma$", "N"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)

phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

phonon.write_yaml_band_structure(filename="{}_{}.yaml".format(os.path.splitext(model_name)[0], config_type))
