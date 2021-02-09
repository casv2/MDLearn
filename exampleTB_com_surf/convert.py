import quippy
from ase.io import read, write
import sys

i = sys.argv[1]

calculator = quippy.Potential("TB NRL-TB", param_filename="/Users/Cas/.julia/dev/MDLearn/exampleTB_com_surf/quip_params.xml")

at = read("/Users/Cas/.julia/dev/MDLearn/exampleTB_com_surf/crash_{}.xyz".format(i))
at.set_calculator(calculator)

at.arrays["force"] = at.get_forces()
at.info["energy"] = at.get_potential_energy()
at.info["config_type"] = "HMD_iter{}".format(i)

write("/Users/Cas/.julia/dev/MDLearn/exampleTB_com_surf/crash_conv_{}.xyz".format(i), at)

i0 = int(i) - 1

al = read("/Users/Cas/.julia/dev/MDLearn/exampleTB_com_surf/DB_{}.xyz".format(i0), ":")
al.append(at)
write("/Users/Cas/.julia/dev/MDLearn/exampleTB_com_surf/DB_{}.xyz".format(i), al)
