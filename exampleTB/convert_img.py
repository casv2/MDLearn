import quippy
from ase.io import read, write
import sys

calculator = quippy.Potential("TB NRL-TB", param_filename="/Users/Cas/.julia/dev/MDLearn/exampleTB/quip_params.xml")

al = read("/Users/Cas/.julia/dev/MDLearn/exampleTB/img_all.xyz", ":")[::10]

print(len(al))

for (i,at) in enumerate(al):
    print(i)
    at.set_calculator(calculator)
    at.arrays["force"] = at.get_forces()
    at.info["energy"] = at.get_potential_energy()
    at.info["config_type"] = "img"

write("/Users/Cas/.julia/dev/MDLearn/exampleTB/img_all_conv.xyz", al)

