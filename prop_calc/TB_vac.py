from quippy.potential import Potential
calculator = Potential("TB NRL-TB", param_filename="./quip_params.xml")

import utilities
import vacancy

from ase.io import read

at = read("NRLTB_hcp3.xyz")
at.rattle(0.001)
at.set_calculator(calculator)
properties = vacancy.do_one_vacancy(at, calculator)

print(properties[4])
