import sys
import os
from ase.io import read, write
import numpy as np
from ase.calculators.castep import Castep

i = sys.argv[1]
config_type = sys.argv[2]

mpirun = "mpirun"
mpirun_args = "-n 64"
castep = "castep.mpi"

os.environ['CASTEP_COMMAND'] = '{0} {1} {2}'.format(mpirun, mpirun_args, castep)

print os.environ['CASTEP_COMMAND']

calculator = Castep(directory="./_CASTEP",
                        cut_off_energy=700, #700
                        max_scf_cycles=250,
                        calculate_stress=True,
                        finite_basis_corr='automatic',
                        smearing_width='0.1',
                        #elec_energy_tol=1E-7,
                        #elec_force_tol=1E-3,
                        fine_grid_scale=3,
                        mixing_scheme='Pulay',
                        kpoints_mp_spacing='0.04', #0.015
                        write_checkpoint='none')


at = read("/Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/crash_{}.xyz".format(i))
at.set_calculator(calculator)

at.arrays["force"] = at.get_forces()
at.info["energy"] = at.get_potential_energy()
#at.info["virial"] = -1.0 * at.get_volume() * at.get_stress(voigt=False)
at.info["config_type"] = "HMD_{}".format(config_type)

write("/Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/crash_conv_{}.xyz".format(i), at)

i0 = int(i) - 1

al = read("/Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/DB_{}.xyz".format(i0), ":")
al.append(at)
write("/Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/DB_{}.xyz".format(i), al)
