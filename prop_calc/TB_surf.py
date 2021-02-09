from quippy.potential import Potential
calculator = Potential("TB NRL-TB", param_filename="./quip_params.xml")

import utilities
import surface

from ase.io import read

surf = read("NRLTB_hcp_surf.xyz")
bulk = read("NRLTB_hcp3.xyz")

bulk.set_calculator(calculator)
surf.set_calculator(calculator)

properties = surface.do_symmetric_surface(bulk, surf, calculator)

print(properties)
print(properties["Ef"])
