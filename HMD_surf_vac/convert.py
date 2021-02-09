import quippy
from ase.io import read, write
import sys

i = sys.argv[1]
config_type = sys.argv[2]

calculator = quippy.Potential("TB NRL-TB", param_filename="/Users/Cas/.julia/dev/MDLearn/HMD_surf_vac/quip_params.xml")

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
